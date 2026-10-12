package ops

// Generated on 2026-10-11 as golden_test.go describes. Do not edit: a changed
// hash is a numerics change (AGENTS.md). Each entry is the hash at
// tensor.SetWorkers(1) and SetConvWorkers(0), then at tensor.SetWorkers(4) and SetConvWorkers(3).

// goldenWhistle holds what go-whistle internal/ops @ 8926ee3 gave for
// commonGoldenCases and whistleGoldenCases, by goldenConfig and case, except
// Conv2D on arm64/neon, arm64/go and wasm/go: from v0.3.0 on (PLAN 1.3,
// regenerated on 2026-10-12) its groups = 1 convolution runs through
// Conv2DPlan there, within KernelTolerance("conv2d") of go-whistle's bits.
// On amd64 Conv2D still gives go-whistle's bits.
var goldenWhistle = map[string]map[string][2]string{
	"amd64/avx2": {
		"Attention":     {"553e9a074dce474a", "553e9a074dce474a"},
		"CausalMask":    {"b793ce8dc7b262c8", "b793ce8dc7b262c8"},
		"Conv1D":        {"e5446abd3b7c2eb5", "e5446abd3b7c2eb5"},
		"Conv1DLeftPad": {"4d598e85c5edefa2", "4d598e85c5edefa2"},
		"Conv2D":        {"21a097322acd1907", "21a097322acd1907"},
		"RoPE":          {"430e19c1a47fa3db", "430e19c1a47fa3db"},
	},
	"amd64/go": {
		"Attention":     {"9d29c8f51fae7955", "9d29c8f51fae7955"},
		"CausalMask":    {"b793ce8dc7b262c8", "b793ce8dc7b262c8"},
		"Conv1D":        {"0cac612261e626e9", "0cac612261e626e9"},
		"Conv1DLeftPad": {"ef3218141f4a266f", "ef3218141f4a266f"},
		"Conv2D":        {"21a097322acd1907", "21a097322acd1907"},
		"RoPE":          {"430e19c1a47fa3db", "430e19c1a47fa3db"},
	},
	"arm64/go": {
		"Attention":     {"8030f02dd8fc987d", "8030f02dd8fc987d"},
		"CausalMask":    {"b793ce8dc7b262c8", "b793ce8dc7b262c8"},
		"Conv1D":        {"adbfda064ee2ab08", "adbfda064ee2ab08"},
		"Conv1DLeftPad": {"0de905657ea47748", "0de905657ea47748"},
		"Conv2D":        {"ca4adf10951d3251", "ca4adf10951d3251"},
		"RoPE":          {"2f722fc48112d1c6", "2f722fc48112d1c6"},
	},
	"arm64/neon": {
		"Attention":     {"7ca04f35ecdf2ae4", "7ca04f35ecdf2ae4"},
		"CausalMask":    {"b793ce8dc7b262c8", "b793ce8dc7b262c8"},
		"Conv1D":        {"399ac9bfc51c0f73", "399ac9bfc51c0f73"},
		"Conv1DLeftPad": {"c1acc97b0ad9134f", "c1acc97b0ad9134f"},
		"Conv2D":        {"ca4adf10951d3251", "ca4adf10951d3251"},
		"RoPE":          {"2f722fc48112d1c6", "2f722fc48112d1c6"},
	},
	"wasm/go": {
		"Attention":     {"9d29c8f51fae7955", "9d29c8f51fae7955"},
		"CausalMask":    {"b793ce8dc7b262c8", "b793ce8dc7b262c8"},
		"Conv1D":        {"0cac612261e626e9", "0cac612261e626e9"},
		"Conv1DLeftPad": {"ef3218141f4a266f", "ef3218141f4a266f"},
		"Conv2D":        {"d6b681f444405320", "d6b681f444405320"},
		"RoPE":          {"430e19c1a47fa3db", "430e19c1a47fa3db"},
	},
}

// goldenPocket holds what go-pocket-tts internal/runtime/ops @ 9faba38 gave
// for commonGoldenCases and pocketGoldenCases, by goldenConfig and case. It has
// no purego tag, so there are no arm64/go and amd64/go entries.
var goldenPocket = map[string]map[string][2]string{
	"amd64/avx2": {
		"Attention":                {"553e9a074dce474a", "553e9a074dce474a"},
		"AttentionWithPositions":   {"01f7282f37a410b0", "01f7282f37a410b0"},
		"CausalMask":               {"b793ce8dc7b262c8", "b793ce8dc7b262c8"},
		"Conv1D":                   {"e5446abd3b7c2eb5", "e5446abd3b7c2eb5"},
		"Conv1DLeftPad":            {"4d598e85c5edefa2", "4d598e85c5edefa2"},
		"ConvTranspose1D":          {"6a780eb286f8164a", "6a780eb286f8164a"},
		"ConvTranspose1DPrePacked": {"3650d5723a7c4d69", "3650d5723a7c4d69"},
		"MLP":                      {"c7f2af5038f24b19", "c7f2af5038f24b19"},
		"RoPE":                     {"430e19c1a47fa3db", "430e19c1a47fa3db"},
	},
	"arm64/neon": {
		"Attention":                {"7ca04f35ecdf2ae4", "7ca04f35ecdf2ae4"},
		"AttentionWithPositions":   {"82c5d4b7c24b0bc5", "82c5d4b7c24b0bc5"},
		"CausalMask":               {"b793ce8dc7b262c8", "b793ce8dc7b262c8"},
		"Conv1D":                   {"399ac9bfc51c0f73", "399ac9bfc51c0f73"},
		"Conv1DLeftPad":            {"c1acc97b0ad9134f", "c1acc97b0ad9134f"},
		"ConvTranspose1D":          {"69556108a7888faa", "be541368de6b3ac7"},
		"ConvTranspose1DPrePacked": {"082e012a6f94e3cb", "7453390298e76552"},
		"MLP":                      {"1832fac943d12c18", "1832fac943d12c18"},
		"RoPE":                     {"2f722fc48112d1c6", "2f722fc48112d1c6"},
	},
	"wasm/go": {
		"Attention":                {"9d29c8f51fae7955", "9d29c8f51fae7955"},
		"AttentionWithPositions":   {"f19c793b878334b8", "f19c793b878334b8"},
		"CausalMask":               {"b793ce8dc7b262c8", "b793ce8dc7b262c8"},
		"Conv1D":                   {"0cac612261e626e9", "0cac612261e626e9"},
		"Conv1DLeftPad":            {"ef3218141f4a266f", "ef3218141f4a266f"},
		"ConvTranspose1D":          {"c30086235170997f", "c30086235170997f"},
		"ConvTranspose1DPrePacked": {"9e745aadcff5d746", "9e745aadcff5d746"},
		"MLP":                      {"a982ebb3bf85a950", "a982ebb3bf85a950"},
		"RoPE":                     {"430e19c1a47fa3db", "430e19c1a47fa3db"},
	},
}

// goldenTranscribe holds what algo-tensor gave for transcribeGoldenCases,
// by goldenConfig and case.
var goldenTranscribe = map[string]map[string][2]string{
	"amd64/go": {
		"Conv2DPlan": {"7d15cc619b8323f9", "7d15cc619b8323f9"},
	},
	"arm64/go": {
		"Conv2DPlan": {"172a537a4961ecaf", "172a537a4961ecaf"},
	},
	"arm64/neon": {
		"Conv2DPlan": {"db0a8e530a94167d", "db0a8e530a94167d"},
	},
	"wasm/go": {
		"Conv2DPlan": {"a58c28226e46f241", "a58c28226e46f241"},
	},
}
