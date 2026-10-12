//go:build amd64 && !purego

// algo-tensor's own, not in go-whistle or go-pocket-tts: dotF32AVX2
// (dot_amd64.s) for three x rows against one weight row at once.

#include "textflag.h"

// func linearDot3AVX2(x *float32, ldx int, w *float32, n int, y *float32, ldy int)
//
// Computes, for r = 0, 1, 2, the dot product of the n float32 values at
// x + r·ldx and at w, and stores it to y + r·ldy (ldx and ldy count
// float32s). n must be >= 1; the caller guarantees that every row is
// valid for n elements.
//
// ABI0 stack layout (amd64, all args on stack via FP):
//   x    +0(FP)   *float32   8 bytes
//   ldx  +8(FP)   int        8 bytes
//   w   +16(FP)   *float32   8 bytes
//   n   +24(FP)   int        8 bytes
//   y   +32(FP)   *float32   8 bytes
//   ldy +40(FP)   int        8 bytes
//   ───────────────────────────────
//   total argsize: 48 bytes
//
// Register allocation:
//   SI, R8, R9   pointers into the x rows 0, 1, 2
//   DI           pointer into w
//   CX           remaining element count
//   Y0–Y3        row 0's four accumulators, folded into Y0
//   Y4–Y7        row 1's, folded into Y4
//   Y8–Y11       row 2's, folded into Y8
//   Y12          the w chunk, shared by the three rows
//   Y13–Y15      the x chunks of rows 0, 1, 2
//   X1, X5, X9   the scalar-tail accumulators of rows 0, 1, 2
//
// Bit-exactness: every row goes through dotF32AVX2's operations in its
// order. Element p of the 32-wide loop goes to accumulator (p/8)%4; the
// four are folded as (Y0+Y1) + (Y2+Y3), only after that loop ran (all four
// are +0 otherwise, so the fold would give +0 too); the 8-wide loop
// continues in the folded accumulator; the scalar tail of 0–7 elements
// accumulates in a register of its own that starts at +0; then
// VEXTRACTF128 + VADDPS (lower + upper), two VHADDPS, and VADDSS of the
// tail. The FMAs keep dotF32AVX2's operands in place as well: the x value
// is the second source (dotF32AVX2's a, loaded into a register) and the
// w value the third (its b, from memory there, from Y12/X12 here). The
// product is commutative, but which NaN an FMA propagates when both
// factors are NaN depends on their order, so it stays the same.
//
// Gotcha, as in dot_amd64.s: VEX-encoded scalar ops (VFMADD231SS, VMOVSS
// from memory) zero bits 255:128 of their destination, so the tails must
// not accumulate into the vector accumulators.

TEXT ·linearDot3AVX2(SB), NOSPLIT, $0-48
    MOVQ x+0(FP),   SI      // SI = x row 0
    MOVQ ldx+8(FP), AX
    SHLQ $2, AX             // AX = ldx in bytes
    LEAQ (SI)(AX*1), R8     // R8 = x row 1
    LEAQ (R8)(AX*1), R9     // R9 = x row 2
    MOVQ w+16(FP),  DI      // DI = w
    MOVQ n+24(FP),  CX      // CX = n

    VPXOR Y0, Y0, Y0
    VPXOR Y1, Y1, Y1
    VPXOR Y2, Y2, Y2
    VPXOR Y3, Y3, Y3
    VPXOR Y4, Y4, Y4
    VPXOR Y5, Y5, Y5
    VPXOR Y6, Y6, Y6
    VPXOR Y7, Y7, Y7
    VPXOR Y8, Y8, Y8
    VPXOR Y9, Y9, Y9
    VPXOR Y10, Y10, Y10
    VPXOR Y11, Y11, Y11

    // ── Main loop: 32 floats (4×8) per iteration ──────────────────────────
    // VFMADD231PS Y12, Y13, Y0  →  Y0 += Y13 * Y12  (x · w, AT&T order)
    CMPQ CX, $32
    JL   tail8

loop32:
    VMOVUPS     0(DI), Y12
    VMOVUPS     0(SI), Y13
    VFMADD231PS Y12, Y13, Y0
    VMOVUPS     0(R8), Y14
    VFMADD231PS Y12, Y14, Y4
    VMOVUPS     0(R9), Y15
    VFMADD231PS Y12, Y15, Y8

    VMOVUPS     32(DI), Y12
    VMOVUPS     32(SI), Y13
    VFMADD231PS Y12, Y13, Y1
    VMOVUPS     32(R8), Y14
    VFMADD231PS Y12, Y14, Y5
    VMOVUPS     32(R9), Y15
    VFMADD231PS Y12, Y15, Y9

    VMOVUPS     64(DI), Y12
    VMOVUPS     64(SI), Y13
    VFMADD231PS Y12, Y13, Y2
    VMOVUPS     64(R8), Y14
    VFMADD231PS Y12, Y14, Y6
    VMOVUPS     64(R9), Y15
    VFMADD231PS Y12, Y15, Y10

    VMOVUPS     96(DI), Y12
    VMOVUPS     96(SI), Y13
    VFMADD231PS Y12, Y13, Y3
    VMOVUPS     96(R8), Y14
    VFMADD231PS Y12, Y14, Y7
    VMOVUPS     96(R9), Y15
    VFMADD231PS Y12, Y15, Y11

    ADDQ $128, SI
    ADDQ $128, R8
    ADDQ $128, R9
    ADDQ $128, DI
    SUBQ  $32, CX
    CMPQ  CX, $32
    JGE  loop32

    // Fold each row's four accumulators: (Y0+Y1) + (Y2+Y3).
    VADDPS Y1, Y0, Y0
    VADDPS Y3, Y2, Y2
    VADDPS Y2, Y0, Y0
    VADDPS Y5, Y4, Y4
    VADDPS Y7, Y6, Y6
    VADDPS Y6, Y4, Y4
    VADDPS Y9, Y8, Y8
    VADDPS Y11, Y10, Y10
    VADDPS Y10, Y8, Y8

    // ── Single-step loop: 8 floats per iteration ──────────────────────────
tail8:
    CMPQ CX, $8
    JL   tail1

loop8:
    VMOVUPS     0(DI), Y12
    VMOVUPS     0(SI), Y13
    VFMADD231PS Y12, Y13, Y0
    VMOVUPS     0(R8), Y14
    VFMADD231PS Y12, Y14, Y4
    VMOVUPS     0(R9), Y15
    VFMADD231PS Y12, Y15, Y8
    ADDQ $32, SI
    ADDQ $32, R8
    ADDQ $32, R9
    ADDQ $32, DI
    SUBQ  $8, CX
    CMPQ  CX, $8
    JGE  loop8

    // ── Scalar tail: 0–7 remaining elements, each row in its own X ────────
tail1:
    VXORPS X1, X1, X1
    VXORPS X5, X5, X5
    VXORPS X9, X9, X9
    TESTQ CX, CX
    JZ    reduce

loop1:
    VMOVSS      0(DI), X12
    VMOVSS      0(SI), X13
    VFMADD231SS X12, X13, X1    // X1[31:0] += x0 · w
    VMOVSS      0(R8), X14
    VFMADD231SS X12, X14, X5
    VMOVSS      0(R9), X15
    VFMADD231SS X12, X15, X9
    ADDQ $4, SI
    ADDQ $4, R8
    ADDQ $4, R9
    ADDQ $4, DI
    DECQ CX
    JNZ  loop1

    // ── Horizontal reduce of each row, as in dotF32AVX2 ───────────────────
reduce:
    VEXTRACTF128 $1, Y0, X2     // upper 4 floats
    VADDPS X2, X0, X0           // lower + upper
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0
    VADDSS X1, X0, X0           // + scalar tail

    VEXTRACTF128 $1, Y4, X6
    VADDPS X6, X4, X4
    VHADDPS X4, X4, X4
    VHADDPS X4, X4, X4
    VADDSS X5, X4, X4

    VEXTRACTF128 $1, Y8, X10
    VADDPS X10, X8, X8
    VHADDPS X8, X8, X8
    VHADDPS X8, X8, X8
    VADDSS X9, X8, X8

    MOVQ y+32(FP),   DX
    MOVQ ldy+40(FP), BX
    SHLQ $2, BX                 // BX = ldy in bytes
    VMOVSS X0, (DX)
    VMOVSS X4, (DX)(BX*1)
    VMOVSS X8, (DX)(BX*2)

    VZEROUPPER                  // clear upper YMM bits before returning to SSE
    RET
