#include "graphics/host_gpu/renderer/pipeline/subgroupSize.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics;

constexpr uint32_t kStageVertex   = 0x00000001u;
constexpr uint32_t kStageTessCtrl = 0x00000002u;
constexpr uint32_t kStageTessEval = 0x00000004u;
constexpr uint32_t kStageFragment = 0x00000010u;
constexpr uint32_t kStageCompute  = 0x00000020u;
constexpr uint32_t kStageTask     = 0x00000040u;
constexpr uint32_t kStageMesh     = 0x00000080u;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "SubgroupSizeTests: failed: %s\n", message);
		std::abort();
	}
}

void TestRadvDualWaveCaps() {
	// RADV on RDNA/GFX12: default=64, min=32, max=64, stages=COMPUTE|FRAGMENT|TASK|MESH.
	const SubgroupSizeCaps radv {
	    .default_size     = 64u,
	    .min_size         = 32u,
	    .max_size         = 64u,
	    .supported_stages = kStageCompute | kStageFragment | kStageTask | kStageMesh,
	};

	// Wave32 fragment shader must request requiredSubgroupSize=32 so RADV does not run it Wave64.
	{
		const auto d = ResolveStageSubgroupSize(kStageFragment, 32u, false, true, radv);
		Check(d.apply && d.required_size == 32u && d.expected_size == 32u && !d.mismatch_warning,
		      "RADV Wave32 fragment requests subgroup size 32");
	}
	// Wave32 fragment shader without subgroup ops also requests Wave32 cleanly.
	{
		const auto d = ResolveStageSubgroupSize(kStageFragment, 32u, false, false, radv);
		Check(d.apply && d.required_size == 32u && !d.mismatch_warning,
		      "RADV Wave32 fragment without subgroup ops");
	}
	// Wave64 fragment shader pins requiredSubgroupSize=64.
	{
		const auto d = ResolveStageSubgroupSize(kStageFragment, 64u, false, true, radv);
		Check(d.apply && d.required_size == 64u && !d.mismatch_warning,
		      "RADV Wave64 fragment requests subgroup size 64");
	}
	// Wave32 mesh shader requests subgroup size 32; Wave64 requests 64.
	{
		const auto d32 = ResolveStageSubgroupSize(kStageMesh, 32u, false, true, radv);
		const auto d64 = ResolveStageSubgroupSize(kStageMesh, 64u, false, true, radv);
		Check(d32.apply && d32.required_size == 32u && !d32.mismatch_warning, "RADV Wave32 mesh");
		Check(d64.apply && d64.required_size == 64u && !d64.mismatch_warning, "RADV Wave64 mesh");
	}
	// Compute Wave32 and Wave64 both apply directly.
	{
		const auto d32 = ResolveStageSubgroupSize(kStageCompute, 32u, false, true, radv);
		const auto d64 = ResolveStageSubgroupSize(kStageCompute, 64u, false, true, radv);
		Check(d32.apply && d32.required_size == 32u && !d32.mismatch_warning,
		      "RADV Wave32 compute");
		Check(d64.apply && d64.required_size == 64u && !d64.mismatch_warning,
		      "RADV Wave64 compute");
	}
	// Vertex/tessellation stages are not in RADV's requiredSubgroupSizeStages:
	// Wave32 with subgroup ops must trigger the permanent mismatch diagnostic.
	{
		const auto d_vs = ResolveStageSubgroupSize(kStageVertex, 32u, false, true, radv);
		Check(!d_vs.apply && d_vs.mismatch_warning && d_vs.expected_size == 32u,
		      "RADV Wave32 vertex with subgroup ops warns");
		const auto d_vs_noop = ResolveStageSubgroupSize(kStageVertex, 32u, false, false, radv);
		Check(!d_vs_noop.apply && !d_vs_noop.mismatch_warning,
		      "RADV Wave32 vertex without subgroup ops stays quiet");
		const auto d_tcs64 = ResolveStageSubgroupSize(kStageTessCtrl, 64u, false, true, radv);
		Check(!d_tcs64.apply && !d_tcs64.mismatch_warning,
		      "RADV Wave64 TCS matches host default 64");
		const auto d_tes32 = ResolveStageSubgroupSize(kStageTessEval, 32u, false, true, radv);
		Check(!d_tes32.apply && d_tes32.mismatch_warning,
		      "RADV Wave32 TES with subgroup ops warns");
	}
}

void TestWave32OnlyGpuCaps() {
	// Wave32-only GPU (e.g., default=32, min=32, max=32, stages=COMPUTE|FRAGMENT).
	const SubgroupSizeCaps wave32_gpu {
	    .default_size     = 32u,
	    .min_size         = 32u,
	    .max_size         = 32u,
	    .supported_stages = kStageCompute | kStageFragment,
	};

	// Emulated Wave64 compute/mesh (lane_count == 2) expects host subgroup size 32.
	{
		const auto d_cs = ResolveStageSubgroupSize(kStageCompute, 64u, true, true, wave32_gpu);
		Check(d_cs.apply && d_cs.required_size == 32u && !d_cs.mismatch_warning,
		      "Wave32 GPU emulated Wave64 compute expects 32");
		const auto d_mesh = ResolveStageSubgroupSize(kStageMesh, 64u, true, true, wave32_gpu);
		Check(!d_mesh.apply && d_mesh.expected_size == 32u && !d_mesh.mismatch_warning,
		      "Wave32 GPU emulated Wave64 mesh matches default 32");
	}
	// Unemulated Wave64 fragment with subgroup ops cannot run at 64 -> warns.
	{
		const auto d_fs = ResolveStageSubgroupSize(kStageFragment, 64u, false, true, wave32_gpu);
		Check(!d_fs.apply && d_fs.mismatch_warning && d_fs.expected_size == 64u,
		      "Wave32 GPU Wave64 fragment with subgroup ops warns");
		const auto d_fs_noop =
		    ResolveStageSubgroupSize(kStageFragment, 64u, false, false, wave32_gpu);
		Check(!d_fs_noop.apply && !d_fs_noop.mismatch_warning,
		      "Wave32 GPU Wave64 fragment without subgroup ops stays quiet");
	}
}

void TestNoControlAndInvalidWave() {
	// Simulates pre-fix state where graphics stages had no subgroupSizeControl.
	const SubgroupSizeCaps compute_only {
	    .default_size     = 64u,
	    .min_size         = 32u,
	    .max_size         = 64u,
	    .supported_stages = kStageCompute,
	};
	{
		const auto d = ResolveStageSubgroupSize(kStageFragment, 32u, false, true, compute_only);
		Check(!d.apply && d.mismatch_warning && d.expected_size == 32u,
		      "Compute-only caps warn on Wave32 fragment with subgroup ops");
	}

	// Driver without subgroupSizeControl at all.
	const SubgroupSizeCaps no_control {
	    .default_size     = 64u,
	    .min_size         = 0u,
	    .max_size         = 0u,
	    .supported_stages = 0u,
	};
	{
		const auto d32 = ResolveStageSubgroupSize(kStageFragment, 32u, false, true, no_control);
		Check(!d32.apply && d32.mismatch_warning, "No control warns on mismatch");
		const auto d64 = ResolveStageSubgroupSize(kStageFragment, 64u, false, true, no_control);
		Check(!d64.apply && !d64.mismatch_warning, "No control matches default 64");
	}

	// Unknown guest wave size is ignored safely.
	{
		const auto d0 = ResolveStageSubgroupSize(kStageFragment, 0u, false, true, compute_only);
		Check(!d0.apply && !d0.mismatch_warning && d0.expected_size == 0u,
		      "Unknown wave size ignored");
	}
}

} // namespace

int main() {
	TestRadvDualWaveCaps();
	TestWave32OnlyGpuCaps();
	TestNoControlAndInvalidWave();
	std::puts("SubgroupSizeTests: ok");
	return 0;
}
