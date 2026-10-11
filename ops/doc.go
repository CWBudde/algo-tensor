// Package ops holds the neural-network operators on top of package tensor:
// scaled dot-product attention with a causal mask and offset or with absolute
// query/key positions and a context window, interleaved RoPE from
// precomputed cos/sin tables, Conv1D (im2col + GEMM for groups=1, a direct
// loop for grouped and depthwise convolutions), ConvTranspose1D, Conv2D in
// NHWC/HWIO with groups (depthwise included) and a two-layer MLP.
//
// From github.com/cwbudde/go-whistle internal/ops @ 8926ee3, which copied
// github.com/cwbudde/go-pocket-tts internal/runtime/ops @ 9faba38 without
// ConvTranspose1D, MLP and the Pocket-TTS position/context attention
// (docs/FINDINGS.md F-008) and added Conv2D and SamePadding. The three are
// back from go-pocket-tts. Each file names its source.
//
// The F-numbers, docs/FINDINGS.md and PLAN.md in this module's comments refer
// to go-whistle's docs/FINDINGS.md and PLAN.md.
package ops
