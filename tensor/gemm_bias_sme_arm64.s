//go:build arm64 && !purego

// algo-tensor's own, not in go-whistle or go-pocket-tts: an SME2 kernel for
// MatMulTransBBias that gives every output gemmNEON4x4's bits (PLAN.md 2.1).

// SME2 GEMM kernel for MatMulTransBBias on Apple M4 and later (streaming
// vector length 512 bits: 16 float32 lanes, four 16×16 float32 ZA tiles).
//
// gemmSME computes c[i*ldc+j] = bias[j] + dot(a[i*k:(i+1)*k],
// b[j*k:(j+1)*k]) for i < m, j < n, with a, b and c in the layout of
// gemmBias (gemm_bias.go), in one streaming-mode section:
//
//  1. It packs the operand with fewer rows whole (16-row panels of a or
//     16-column panels of b) and the other one panel by panel, each just
//     before its blocks run against every panel of the first. A panel
//     holds kp = 4⌈k/4⌉ steps of 16 values, k-major; step p < full =
//     4⌊k/4⌋ is column p of the panel's rows. The tail group (k%4 > 0),
//     steps full..full+3, replays gemmNEON4x4's reload of columns k-4..k-1
//     with the already-summed lanes zeroed in both operands: step full+s is
//     column k-4+s for s >= 4-k%4 and +0 below. Rows past m or n are +0.
//     The transpose goes through tile za0: rows are loaded as horizontal
//     slices and stored as vertical ones, so packing touches no Z register.
//  2. For every 16×16 output block, ZA tile t accumulates the outer products
//     of the steps p ≡ t (mod 4) from +0, as lane t of gemmNEON4x4's
//     accumulators does; FMOPA is fused like FMLA, and a +0 step adds
//     fma(+0, +0, acc) as the masked NEON lanes do.
//  3. The four tiles are reduced as REDUCE4's FADDP tree does,
//     ((t0 + t1) + (t2 + t3)) + bias, and stored into c with stride ldc,
//     predicated to the block's columns and rows.
//
// NaNs. FMOPA, like every instruction that accumulates into ZA, behaves as
// if FPCR.DN were set: a NaN it produces is the default NaN, where FMLA
// passes on the payload and sign of a NaN operand. Outputs that are not
// NaN have gemmNEON4x4's bits. z28 collects 0·output over all outputs,
// which is NaN for a NaN or infinite one; the kernel tests it once at the
// end and reports whether it found one, and the caller then computes the
// outputs again with gemmNEON4x4. The collector costs 4-7 %; a test per
// block, or per row, made the canary checks wait for the chain of FMLAs
// into z28 and cost 30-40 %. A signal also clears lanes of z28, but every
// signal fails the next canary check, here or once more at the end, and
// after a failed check the caller looks for NaN outputs itself.
//
// Signals. When a signal handler returns to a thread in streaming mode,
// Darwin zeroes lanes 4-15 of every Z register; ZA, the P registers and the
// general registers survive (PLAN.md Phase 2). Go's preemption and profiling
// signals arrive at any time, so every block is guarded by a canary: z31 is
// set to all ones at the start and compared with zero after every block's
// k loop and after its stores. If any lane is zero, a signal may have
// clobbered an operand since the last check: after the k loop the block is
// redone from a zeroed ZA, after the stores (ZA is intact) only the
// reduction is redone, and the canary is set again. After 8 redos of one
// block the kernel gives up and returns the outer rows it finished, and the
// caller computes the rest with gemmNEON4x4, which gives the same bits.
//
// Go's assembler has no SME mnemonics: the instructions are WORDs, each
// with its mnemonic, taken from `xcrun clang -c -mcpu=apple-m4
// -march=armv9.2-a+sme2` and `xcrun objdump -d`. Registers are fixed in the
// encodings; the comments name them.
//
// Registers R18 (platform), R26 (closure context), R27 (assembler temp),
// R28 (g), R29 (frame pointer) and R30 (link register) are left alone.

#include "textflag.h"

#define SMSTART     WORD $0xd503477f // smstart
#define SMSTOP      WORD $0xd503467f // smstop
#define PTRUE_P0    WORD $0x2598e3e0 // ptrue p0.s
#define PTRUE_PN8   WORD $0x25a07810 // ptrue pn8.s
#define ZERO_ZA     WORD $0xc00800ff // zero {za}
#define ZERO_ZA0    WORD $0xc0080011 // zero {za0.s}
#define CANARY      WORD $0x25b8dfff // mov z31.s, #-1
#define ZERO_Z27    WORD $0x25b8c01b // mov z27.s, #0
#define ZERO_Z28    WORD $0x25b8c01c // mov z28.s, #0
#define CANARY_ZERO WORD $0x258083e1 // cmpeq p1.s, p0/z, z31.s, #0 (Z clear: some lane is zero)
#define NAN_P5      WORD $0x659cc385 // fcmuo p5.s, p0/z, z28.s, z28.s
#define PTEST_P5    WORD $0x2550c0a0 // ptest p0, p5.b (Z clear: an output was NaN or infinite)

// Test hook (smeClobberK, smeClobberR): zero lane 7 of the canary and the
// NaN collector z28, as a signal zeroes lanes 4-15 of both.
#define CLOBBER \
	WORD $0x04a1401d; /* index z29.s, #0, #1 */ \
	WORD $0x258783a3; /* cmpeq p3.s, p0/z, z29.s, #7 */ \
	WORD $0x0593401f; /* mov z31.s, p3/m, #0 */ \
	ZERO_Z28

// Packing.
#define WHILELT_P1_X13_X7  WORD $0x25a715a1 // whilelt p1.s, x13, x7
#define WHILELT_P1_XZR_X15 WORD $0x25af17e1 // whilelt p1.s, xzr, x15
#define LD1W_ZA0H          WORD $0xe09f05c0 // ld1w {za0h.s[w12, 0]}, p1/z, [x14]
#define ST1W_ZA0V          WORD $0xe0bf81c0 // st1w {za0v.s[w12, 0]}, p0, [x14]

// The k loop.
#define LD4_A WORD $0xa040c2c0 // ld1w {z0.s-z3.s}, pn8/z, [x22]
#define LD4_B WORD $0xa040c2e4 // ld1w {z4.s-z7.s}, pn8/z, [x23]
#define FMOPA4 \
	WORD $0x80840000; /* fmopa za0.s, p0/m, p0/m, z0.s, z4.s */ \
	WORD $0x80850021; /* fmopa za1.s, p0/m, p0/m, z1.s, z5.s */ \
	WORD $0x80860042; /* fmopa za2.s, p0/m, p0/m, z2.s, z6.s */ \
	WORD $0x80870063  /* fmopa za3.s, p0/m, p0/m, z3.s, z7.s */

// The reduction.
#define WHILELT_P2_XZR_X14 WORD $0x25ae17e2 // whilelt p2.s, xzr, x14
#define LD1W_BIAS          WORD $0xa540a9fe // ld1w {z30.s}, p2/z, [x15]
#define REDUCE_ROW \
	WORD $0xc0820000; /* mov z0.s, p0/m, za0h.s[w12, 0] */ \
	WORD $0xc0820081; /* mov z1.s, p0/m, za1h.s[w12, 0] */ \
	WORD $0xc0820102; /* mov z2.s, p0/m, za2h.s[w12, 0] */ \
	WORD $0xc0820183; /* mov z3.s, p0/m, za3h.s[w12, 0] */ \
	WORD $0x65810000; /* fadd z0.s, z0.s, z1.s */ \
	WORD $0x65830042; /* fadd z2.s, z2.s, z3.s */ \
	WORD $0x65820000; /* fadd z0.s, z0.s, z2.s */ \
	WORD $0x659e0000; /* fadd z0.s, z0.s, z30.s */ \
	WORD $0xe540e9a0; /* st1w {z0.s}, p2, [x13] */ \
	WORD $0x65bb081c  /* fmla z28.s, p2/m, z0.s, z27.s: z28 += 0·output, NaN for a NaN or infinite output */

// func gemmSME(c *float32, ldc int, a, b, bias *float32, m, n, k int, po, pi *float32, bOuter, clobberK, clobberR int) (redos, done, nan int)
//
// m >= 1, n a positive multiple of 4, k >= 4, ldc >= n. The inner operand,
// b (bOuter = 0) or a (bOuter = 1), is packed whole into pi, 16⌈rows/16⌉·kp
// floats; the outer one is packed one 16-row panel at a time into po,
// 16·kp floats, just before its blocks run against every inner panel. So
// the packed inner operand is the one read again and again, and the outer
// one is read once. clobberK and clobberR are test hooks, 0 in production:
// from<<32 | count clobbers the canary before count checks after the k
// loop (after the stores), from the check numbered from on. redos counts
// the failed checks: each redoes a block or its stores, except a failure
// of the last check, after all blocks. done is the number of outer rows
// (rows of a for bOuter = 0, of b for 1) finished: all of them, or fewer
// when a block failed its check 9 times. nan is 1 if an output stored was
// NaN or infinite (or garbage from a signal, before its redo), else 0;
// after a failed check, a NaN output may go unreported.
TEXT ·gemmSME(SB), NOSPLIT, $0-128
	MOVD c+0(FP), R0
	MOVD ldc+8(FP), R1
	MOVD a+16(FP), R2
	MOVD b+24(FP), R12
	MOVD bias+32(FP), R4
	MOVD m+40(FP), R5
	MOVD n+48(FP), R6
	MOVD k+56(FP), R7
	MOVD po+64(FP), R8
	MOVD pi+72(FP), R9
	MOVD bOuter+80(FP), R3
	MOVD clobberK+88(FP), R10
	MOVD clobberR+96(FP), R11

	LSL  $2, R1, R1    // R1 = ldc*4, byte stride of the rows of c
	LSL  $2, R7, R17   // R17 = k*4, byte stride of the rows of a and b
	ADD  $3, R7, R16
	AND  $-4, R16, R16 // R16 = kp, steps per panel
	AND  $-4, R7, R7   // R7 = full, the columns before the tail group
	MOVD $0, R19       // R19 = failed checks

	SMSTART
	PTRUE_P0
	PTRUE_PN8
	CANARY
	ZERO_Z27
	ZERO_Z28

	// Pack the inner operand into pi; R2 = the outer one.
	CBZ  R3, inner_b
	MOVD R2, R21
	MOVD R5, R22
	MOVD R12, R2
	B    inner_pack

inner_b:
	MOVD R12, R21
	MOVD R6, R22

inner_pack:
	MOVD R9, R23
	MOVD $0, R24
	B    pack

packed_inner:
	MOVD $0, R20 // R20 = o0, first outer row of the panel

outer:
	// Pack outer rows o0..o0+15 into po.
	MUL  R20, R17, R21
	ADD  R2, R21, R21
	CMP  $0, R3
	CSEL EQ, R5, R6, R22 // R22 = outer rows, m or n
	SUB  R20, R22, R22
	MOVD $16, R24
	CMP  R24, R22
	CSEL GT, R24, R22, R22 // R22 = min(16, rows-o0)
	MOVD R8, R23
	MOVD $1, R24
	B    pack

packed_outer:
	MOVD $0, R21 // R21 = first inner row of the block

inner:
	MOVD $0, R25 // R25 = failed checks of this block

block:
	ZERO_ZA
	MUL  R21, R16, R24
	ADD  R24<<2, R9, R24 // R24 = inner panel, pi + i*kp*4
	CBZ  R3, block_a
	MOVD R24, R22        // R22 = A panel
	MOVD R8, R23         // R23 = B panel
	B    block_k

block_a:
	MOVD R8, R22
	MOVD R24, R23

block_k:
	LSR $2, R16, R24 // R24 = kp/4 groups of four steps

kloop:
	LD4_A  // z0..z3 = A steps p..p+3
	LD4_B  // z4..z7 = B steps p..p+3
	ADD    $256, R22
	ADD    $256, R23
	FMOPA4 // tile t += step p+t
	SUB    $1, R24
	CBNZ   R24, kloop

	CBZ  R10, checkk
	LSR  $32, R10, R24
	CBZ  R24, clobberk
	MOVD $0x100000000, R24
	SUB  R24, R10, R10 // not yet: one check fewer to go
	B    checkk

clobberk:
	SUB $1, R10, R10
	CLOBBER

checkk:
	CANARY_ZERO
	BNE redok

reduce:
	CBZ  R3, reduce_a
	MOVD R21, R22 // R22 = i0
	MOVD R20, R23 // R23 = j0
	B    reduce_ij

reduce_a:
	MOVD R20, R22
	MOVD R21, R23

reduce_ij:
	SUB  R23, R6, R14
	WHILELT_P2_XZR_X14     // p2 = columns j0+l < n
	ADD  R23<<2, R4, R15
	LD1W_BIAS              // z30 = bias[j0:j0+16]
	MUL  R22, R1, R13
	ADD  R0, R13, R13
	ADD  R23<<2, R13, R13  // R13 = &c[i0*ldc+j0]
	SUB  R22, R5, R14
	MOVD $16, R15
	CMP  R15, R14
	CSEL GT, R15, R14, R14 // R14 = rows, min(16, m-i0)
	MOVW $0, R12           // w12 = row of the block

rloop:
	REDUCE_ROW
	ADD  R1, R13, R13
	ADD  $1, R12
	SUB  $1, R14
	CBNZ R14, rloop

	CBZ  R11, checkr
	LSR  $32, R11, R24
	CBZ  R24, clobberr
	MOVD $0x100000000, R24
	SUB  R24, R11, R11
	B    checkr

clobberr:
	SUB $1, R11, R11
	CLOBBER

checkr:
	CANARY_ZERO
	BNE redor

	ADD  $16, R21
	CMP  $0, R3
	CSEL EQ, R6, R5, R24 // R24 = inner rows, n or m
	CMP  R24, R21
	BLT  inner

	ADD  $16, R20
	CMP  $0, R3
	CSEL EQ, R5, R6, R24 // R24 = outer rows, m or n
	CMP  R24, R20
	BLT  outer

	// A signal since the last check may have clobbered z28: count it as a
	// redo, so the caller checks the outputs itself.
	NAN_P5
	CANARY_ZERO
	BEQ  done_ok
	ADD  $1, R19

done_ok:
	PTEST_P5
	CSET NE, R12
	SMSTOP
	MOVD R19, redos+104(FP)
	MOVD R24, done+112(FP)
	MOVD R12, nan+120(FP)
	RET

redok:
	// A signal hit the k loop: redo the block from a zeroed ZA.
	ADD $1, R19
	ADD $1, R25
	CMP $8, R25
	BGT bail
	CANARY
	B   block

redor:
	// A signal hit the reduction: ZA is intact, redo the stores.
	ADD $1, R19
	ADD $1, R25
	CMP $8, R25
	BGT bail
	CANARY
	B   reduce

bail:
	NAN_P5
	PTEST_P5
	CSET NE, R12
	SMSTOP
	MOVD R19, redos+104(FP)
	MOVD R20, done+112(FP)
	MOVD R12, nan+120(FP)
	RET

	// pack writes the R22 rows at R21 (row stride R17) as panels at R23,
	// one panel per 16 rows, then returns to packed_inner (R24 = 0) or
	// packed_outer. It uses R12..R15, R21..R23 and R25.
pack:
	ZERO_ZA0
	MOVD $16, R25
	CMP  R25, R22
	CSEL LT, R22, R25, R25 // R25 = rows of this panel, min(R22, 16)
	MOVD $0, R13           // R13 = k0, first column of the chunk

pack_chunk:
	WHILELT_P1_X13_X7     // p1 = columns k0+l < full
	MOVW $0, R12
	ADD  R13<<2, R21, R14 // R14 = &row 0 [k0]
	MOVD R25, R15

pack_rows:
	LD1W_ZA0H // za0 slice w12 = row w12, columns k0..k0+15
	ADD  R17, R14, R14
	ADD  $1, R12
	SUB  $1, R15
	CBNZ R15, pack_rows

	SUB  R13, R7, R15
	MOVD $16, R12
	CMP  R12, R15
	CSEL GT, R12, R15, R15 // R15 = columns of this chunk, min(16, full-k0)
	MOVW $0, R12
	ADD  R13<<6, R23, R14  // R14 = step k0 of the panel

pack_cols:
	ST1W_ZA0V // step k0+w12 = vertical slice w12
	ADD  $64, R14
	ADD  $1, R12
	SUB  $1, R15
	CBNZ R15, pack_cols

	ADD $16, R13
	CMP R7, R13
	BLT pack_chunk

	// Tail group: load columns full..k-1 into lanes 0..tail-1 (the rest +0)
	// and store step full+s from vertical slice (s+tail)&3, which is column
	// full+s-(4-tail) for s >= 4-tail and +0 below.
	LSR  $2, R17, R15
	SUB  R7, R15, R15    // R15 = tail = k - full
	CBZ  R15, pack_next
	WHILELT_P1_XZR_X15   // p1 = lanes l < tail
	MOVW $0, R12
	ADD  R7<<2, R21, R14 // R14 = &row 0 [full]
	MOVD R25, R13

pack_trows:
	LD1W_ZA0H
	ADD  R17, R14, R14
	ADD  $1, R12
	SUB  $1, R13
	CBNZ R13, pack_trows

	MOVW R15, R12
	ADD  R7<<6, R23, R14 // R14 = step full of the panel
	MOVD $4, R13

pack_tcols:
	ST1W_ZA0V
	ADD  $64, R14
	ADD  $1, R12
	AND  $3, R12
	SUB  $1, R13
	CBNZ R13, pack_tcols

pack_next:
	ADD  R17<<4, R21, R21 // 16 rows on
	ADD  R16<<6, R23, R23 // one panel on: kp steps of 64 bytes
	SUBS $16, R22, R22
	BGT  pack
	CBZ  R24, packed_inner
	B    packed_outer
