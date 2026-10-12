//go:build amd64 && !purego

// algo-tensor's own, not in go-whistle or go-pocket-tts: MatMulTransB on
// amd64 through Linear's kernel (PLAN.md 1.4).

package tensor

// matMulTransB gives every output dotF32(a row, b row), the arithmetic of
// matMulTransBGeneric, so no bit changes; only the order of the outputs
// does. Where dotF32 would take dotF32AVX2 (AVX2 + FMA, k >= 8), it runs
// over panels of b rows that stay in L2, as Linear does, and in each panel
// groups of three a rows over each b row in linearDot3AVX2, which gives
// each output dotF32AVX2's bits; the a rows left over take dotF32.
func matMulTransB(c []float32, ldc int, a, b []float32, m, n, k int) {
	if !useAVX2FMA || k < 8 || m < linearGroup {
		matMulTransBGeneric(c, ldc, a, b, m, n, k)
		return
	}

	panel := linearPanelRows(k)

	for j0 := 0; j0 < n; j0 += panel {
		j1 := min(n, j0+panel)

		i := 0
		for ; i+linearGroup <= m; i += linearGroup {
			as := a[i*k:]
			cs := c[i*ldc:]

			for j := j0; j < j1; j++ {
				linearDot3(cs[j:], ldc, as, k, b[j*k:(j+1)*k])
			}
		}

		for ; i < m; i++ {
			aRow := a[i*k : (i+1)*k]
			cRow := c[i*ldc : i*ldc+n]

			for j := j0; j < j1; j++ {
				cRow[j] = dotF32(aRow, b[j*k:(j+1)*k])
			}
		}
	}
}
