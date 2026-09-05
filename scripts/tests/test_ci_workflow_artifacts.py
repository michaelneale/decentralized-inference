import json
import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"
SLICES = ROOT / "ci/slices.yml"
PLAN_ACTION = ROOT / ".github/actions/plan-ci/action.yml"


class CiWorkflowArtifactTests(unittest.TestCase):
    def test_main_entrypoints_are_focused(self):
        for lane in ("quality", "website", "linux", "macos", "windows"):
            workflow = (WORKFLOWS / f"main_{lane}.yml").read_text()
            self.assertIn("push:", workflow)
            self.assertIn("branches: [main]", workflow)
            self.assertIn(f"uses: ./.github/workflows/ci-{lane}-lane.yml", workflow)
            self.assertNotIn("github.rest.actions.createWorkflowDispatch", workflow)
            self.assertNotIn("linux_host_input:", workflow)

        compatibility = (WORKFLOWS / "ci.yml").read_text()
        self.assertIn("workflow_call:", compatibility)
        self.assertNotIn("\n  push:\n", compatibility)
        self.assertNotIn("uses: ./.github/workflows/ci-", compatibility)

    def test_main_plan_uses_all_rows_and_bounded_budgets(self):
        profile = json.loads(SLICES.read_text())["profiles"]["main"]
        self.assertTrue(profile["all_rows"])
        self.assertEqual(profile["budgets"]["total_max_workers"], 18)
        self.assertEqual(profile["budgets"]["linux_max_parallel"], 12)
        self.assertEqual(profile["budgets"]["macos_max_parallel"], 4)
        self.assertEqual(profile["budgets"]["windows_max_parallel"], 2)

    def test_orchestrator_calls_same_slices_for_pr_and_main(self):
        workflow = "\n".join(
            (WORKFLOWS / f"ci-{lane}-lane.yml").read_text()
            for lane in ("quality", "website", "linux", "macos", "windows")
        )
        for path in (
            "ci-quality-slice.yml",
            "ci-web-slice.yml",
            "ci-ui-artifact-slice.yml",
            "ci-rust-tests-slice.yml",
            "ci-linux-host-slice.yml",
            "ci-macos-host-slice.yml",
            "ci-windows-host-slice.yml",
            "ci-linux-runtime-slice.yml",
            "ci-linux-product-slice.yml",
            "ci-macos-runtime-slice.yml",
            "ci-macos-product-slice.yml",
            "ci-windows-runtime-slice.yml",
            "ci-windows-product-slice.yml",
            "ci-platform-checks-slice.yml",
            "ci-linux-product-smoke-slice.yml",
            "ci-macos-product-smoke-slice.yml",
            "ci-linux-sdk-slice.yml",
            "ci-macos-sdk-slice.yml",
        ):
            self.assertIn(f"uses: ./.github/workflows/{path}", workflow)

    def test_static_abi_is_shared_by_tests_and_sdk(self):
        workflow = (WORKFLOWS / "ci-linux-lane.yml").read_text()
        self.assertIn("artifact_name: ci-static-abi-${{ github.run_id }}", workflow)
        self.assertIn("static_abi_artifact_name: ci-static-abi-${{ github.run_id }}", workflow)

    def test_platform_products_do_not_wait_for_unrelated_platforms(self):
        for platform in ("linux", "macos", "windows"):
            with self.subTest(platform=platform):
                workflow = (WORKFLOWS / f"ci-{platform}-lane.yml").read_text()
                self.assertIn("needs: [hosts, native_runtimes]", workflow)
                for other in ({"linux", "macos", "windows"} - {platform}):
                    self.assertNotIn(f"hosts_{other}", workflow)
                    self.assertNotIn(f"native_runtimes_{other}", workflow)

    def test_sdk_producers_start_before_product_composition(self):
        linux = (WORKFLOWS / "ci-linux-lane.yml").read_text()
        macos = (WORKFLOWS / "ci-macos-lane.yml").read_text()
        kotlin = linux[linux.index("  kotlin_sdk_input:"):linux.index("  sdk:")]
        swift = macos[macos.index("  swift_sdk_input:"):macos.index("  sdk:")]

        self.assertIn("needs: [static_abi]", kotlin)
        self.assertNotIn("runtime_product", kotlin)
        self.assertNotIn("needs:", swift)
        self.assertNotIn("runtime_product", swift)
        self.assertIn(
            "max_parallel: ${{ fromJson(inputs.lane_plan_json).budgets.macos_max_parallel }}",
            swift,
        )
        self.assertIn(
            "fail_fast: ${{ inputs.original_event_name == 'pull_request' }}",
            swift,
        )
        swift_producer = (WORKFLOWS / "swift-sdk-artifact.yml").read_text()
        self.assertIn("fail-fast: ${{ inputs.fail_fast }}", swift_producer)
        release = (WORKFLOWS / "release.yml").read_text()
        release_swift = release[
            release.index("  build_swift_sdk_artifact:"):
            release.index("  publish:")
        ]
        self.assertIn("fail_fast: false", release_swift)
        self.assertIn("needs: [runtime_product, kotlin_sdk_input]", linux)
        self.assertIn(
            "needs: [validate_plan, runtime_product, swift_sdk_input]",
            macos,
        )

    def test_static_abi_toolchain_epoch_flows_to_rust_test_consumer(self):
        orchestrator = (WORKFLOWS / "ci-linux-lane.yml").read_text()
        producer = (WORKFLOWS / "static-abi-artifact.yml").read_text()
        consumer = (WORKFLOWS / "ci-rust-tests-slice.yml").read_text()

        self.assertIn(
            "value: ${{ jobs.static_abi_artifact.outputs.toolchain_epoch }}",
            producer,
        )
        self.assertIn(
            "toolchain_epoch: ${{ steps.native_toolchain.outputs.epoch }}",
            producer,
        )
        self.assertIn(
            "static_abi_toolchain_epoch: ${{ needs.static_abi.outputs.toolchain_epoch }}",
            orchestrator,
        )
        self.assertIn(
            "pinned_epoch: ${{ inputs.static_abi_toolchain_epoch }}",
            consumer,
        )

    def test_safetensors_runtime_smoke_is_fail_closed_and_compiles_once(self):
        workflow = (WORKFLOWS / "ci-rust-tests-slice.yml").read_text()
        smoke = workflow[workflow.index("  safetensors_runtime_smoke:"):]
        test_name = (
            "inference::skippy::resolver::tests::"
            "safetensors_checkpoint_reaches_mesh_host_runtime"
        )

        self.assertIn('MESH_LLM_SKIP_UI: "1"', smoke)
        self.assertIn("uses: ./.github/actions/restore-sccache-seed", smoke)
        self.assertEqual(smoke.count("cargo test --locked"), 1)
        self.assertIn("--lib --no-run --message-format=json", smoke)
        self.assertIn(f'test_name="{test_name}"', smoke)
        self.assertIn('grep -Fqx "$test_name: test"', smoke)
        self.assertIn(
            '"$SAFETENSORS_SMOKE_TEST_NAME" --exact --ignored --nocapture',
            smoke,
        )

    def test_runtime_image_verification_preserves_backend_argument(self):
        workflow = (WORKFLOWS / "ci-linux-runtime-slice.yml").read_text()

        self.assertIn('read -r -a verify_args <<< "$VERIFY_BACKEND"', workflow)
        self.assertIn('verify-runner-image "${verify_args[@]}"', workflow)

    def test_windows_cuda_runtime_declares_the_installed_toolkit_version(self):
        workflow = (WORKFLOWS / "ci-windows-runtime-slice.yml").read_text()

        self.assertIn(
            "WINDOWS_CUDA_VERSION: ${{ vars.CUDA_VERSION || '12.6.3' }}",
            workflow,
        )
        self.assertIn(
            "MESH_CUDA_VERSION: ${{ vars.CUDA_VERSION || '12.6.3' }}",
            workflow,
        )

    def test_cuda_smoke_uses_the_registered_gpu_runner_labels(self):
        product_smoke = (WORKFLOWS / "product-integration-smoke.yml").read_text()

        self.assertIn(
            '["self-hosted","Linux","X64","amd64","gpu-nvidia",'
            '"mesh-llm-amd64","mesh-llm"]',
            product_smoke,
        )
        self.assertIn("inputs.platform == 'linux' && inputs.backend == 'cuda'", product_smoke)
        self.assertIn("cuda-cudart-12-9", product_smoke)
        self.assertIn("libcublas-12-9", product_smoke)

    def test_two_node_split_smoke_covers_dense_and_recurrent_models(self):
        workflow = (WORKFLOWS / "product-integration-smoke.yml").read_text()
        restore = (ROOT / ".github/actions/restore-product-integration-inputs/action.yml").read_text()
        product_script = (ROOT / "scripts/ci-product-integration-smoke.sh").read_text()
        smoke_script = (ROOT / "scripts/ci-two-node-split-smoke.sh").read_text()

        self.assertIn("restore-product-integration-inputs", workflow)
        self.assertIn("smollm2-q8-inference", restore)
        self.assertIn("family-granite-hybrid", restore)
        self.assertIn("--github-output-prefix dense_", restore)
        self.assertIn("--github-output-prefix recurrent_", restore)
        self.assertIn("MESH_TWO_NODE_SPLIT_CLIENT_ROUTING=1", product_script)
        self.assertIn("run_phase dense-split-kv", product_script)
        self.assertIn("run_phase recurrent-split-kv", product_script)
        self.assertIn("MESH_TWO_NODE_SPLIT_EXPECTED_EXACT_PAYLOAD_KIND=kv-recurrent", product_script)
        self.assertNotIn("MESH_TWO_NODE_SPLIT_RECURRENT_MODEL=", product_script)
        self.assertIn("run_client_routing_probe", smoke_script)
        self.assertIn("Passive client routing and streaming validated", smoke_script)
        self.assertIn(
            'checkpointed_restore = exact_payload_kind == "kv-recurrent"',
            smoke_script,
        )
        self.assertIn(
            "if not checkpointed_restore and (", smoke_script
        )

    def test_granite_rollout_retains_qwen_and_defers_accelerator_product_rows(self):
        slices = json.loads(SLICES.read_text())
        smoke_ids = {row["id"] for row in slices["smoke_rows"]}
        linux = (WORKFLOWS / "ci-linux-product-smoke-slice.yml").read_text()

        self.assertIn("qwen-recurrent-gate", smoke_ids)
        self.assertIn("Qwen3.5-0.8B-Q4_K_M.gguf", linux)
        self.assertIn("expected_exact_payload_kind: kv-recurrent", linux)
        self.assertNotIn("product-integration-cuda", smoke_ids)
        self.assertNotIn("product-integration-metal", smoke_ids)
        self.assertIn("core-cuda", smoke_ids)
        self.assertIn("metal-model-load", smoke_ids)

    def test_cuda_product_supports_the_registered_gpu_runner_architecture(self):
        runtimes = json.loads(SLICES.read_text())["runtime_rows"]
        cuda = next(row for row in runtimes if row["id"] == "linux-cuda")

        self.assertEqual(cuda["cuda_architectures"], "86;120")
        self.assertIn("sm86", cuda["build_dir"])
        self.assertIn("sm120", cuda["build_dir"])

    def test_manifest_budgets_drive_lane_parallelism(self):
        action = PLAN_ACTION.read_text()
        workflow = "\n".join(
            (WORKFLOWS / f"ci-{lane}-lane.yml").read_text()
            for lane in ("quality", "linux", "macos", "windows")
        )
        for budget in (
            "linux_max_parallel",
            "macos_max_parallel",
            "windows_max_parallel",
            "total_max_workers",
        ):
            self.assertIn(
                f'echo "{budget}=$(jq -r \'.budgets.{budget}\' ci-plan.json)"',
                action,
            )
            self.assertIn(
                f"fromJson(inputs.lane_plan_json).budgets.{budget}",
                workflow,
            )
        self.assertNotIn("contains(needs.plan.outputs.profile, 'pr-') &&", workflow)

    def test_plan_action_emits_real_platform_matrices_and_optional_affected_crates(self):
        action = PLAN_ACTION.read_text()
        self.assertIn('select(.platform == "linux")', action)
        self.assertIn('--arg signal "$signal"', action)
        self.assertNotIn('select(.platform == \\"linux\\")', action)
        self.assertIn(
            "if ($affected_crates | length) > 0 then {affected_crates: $affected_crates}",
            action,
        )

    def test_ui_cache_and_website_dependencies_are_explicit(self):
        ui = (ROOT / ".github/workflows/ci-ui-artifact-slice.yml").read_text()
        web = (ROOT / ".github/workflows/ci-web-slice.yml").read_text()
        # UI installs point at the runner image's baked pnpm store instead
        # of the Actions cache (#1392) -- there is nothing left here to
        # restore or save.
        # Matches quoted or unquoted `cache: pnpm`/`cache: npm` -- a plain
        # substring check would miss `cache: "pnpm"` / `cache: 'npm'`, which
        # would still enable setup-node's own dependency cache.
        cache_config = re.compile(
            r"(?m)^[ \t]*cache:[ \t]*"
            r"(?:pnpm|npm|'pnpm'|'npm'|\"pnpm\"|\"npm\")"
            r"[ \t]*(?:#.*)?$"
        )
        self.assertIn("run: pnpm config set store-dir /home/runner/.local/share/pnpm/store", ui)
        self.assertNotIn("uses: actions/cache", ui)
        self.assertNotRegex(ui, cache_config)
        self.assertNotIn("CACHE_NAMESPACE", ui)
        self.assertIn("run: pnpm config set store-dir /home/runner/.local/share/pnpm/store", web)
        self.assertNotIn("uses: actions/cache", web)
        self.assertNotRegex(web, cache_config)
        self.assertNotIn("CACHE_NAMESPACE", web)
        # The `website` job runs in the prebuilt public-web image with no
        # bare-metal row, so setup-node's own npm cache and the `just`
        # install-action were deleted outright (both are baked in the
        # image) rather than gated -- unlike ui_quality/ui_e2e above, this
        # job has no native-cache consumer left to assert on.
        self.assertIn("working-directory: website", web)
        self.assertIn("run: npm ci", web)


if __name__ == "__main__":
    unittest.main()
