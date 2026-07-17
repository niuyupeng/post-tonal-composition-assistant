# Project 2 Corrected Full Run Report

- Completion gate: PASS

## Commands Executed
- `.\.venv311\Scripts\python.exe --version`
- `.\.venv311\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"`
- `.\.venv311\Scripts\python.exe -m pytest -q`
- `.\scripts\smoke_project2.ps1`
- `.\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs/post_tonal_rule_baseline.yaml --split test --experiment-name rule_baseline --output results/project2_v3_rule_baseline_metrics.json --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --examples-output results/project2_v3_generation_examples.json`
- `.\.venv311\Scripts\python.exe -u -m post_tonal.train --config configs/post_tonal_main.yaml --auto-oom-retry --resume`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run env-check`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m pytest`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run write-split-summary --config configs/post_tonal_main.yaml --output results/project2_v3_full_split_summary.json`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs/post_tonal_main.yaml --split test --experiment-name proposed_constraint_guided_transformer --output C:\Users\nyp\Documents\yinyue2\results\project2_v3_proposed_constraint_guided_transformer_metrics.json --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --examples-output results/project2_v3_generation_examples.json --export-dir results/eval_musicxml_v3/proposed_constraint_guided_transformer --checkpoint C:\Users\nyp\Documents\yinyue2\runs\v3\proposed_constraint_guided_transformer\checkpoint.pt`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs/post_tonal_transformer_vanilla.yaml --split test --experiment-name vanilla_transformer --output C:\Users\nyp\Documents\yinyue2\results\project2_v3_vanilla_transformer_metrics.json --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --examples-output results/project2_v3_generation_examples.json --export-dir results/eval_musicxml_v3/vanilla_transformer --checkpoint C:\Users\nyp\Documents\yinyue2\runs\v3\vanilla_transformer\checkpoint.pt`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.train --config configs/post_tonal_transformer_no_constraints.yaml --auto-oom-retry --resume`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs/post_tonal_transformer_no_constraints.yaml --split test --experiment-name transformer_no_constraints --output C:\Users\nyp\Documents\yinyue2\results\project2_v3_transformer_no_constraints_metrics.json --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --examples-output results/project2_v3_generation_examples.json --export-dir results/eval_musicxml_v3/transformer_no_constraints --checkpoint C:\Users\nyp\Documents\yinyue2\runs\v3\transformer_no_constraints\checkpoint.pt`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.train --config configs/post_tonal_without_pcset_constraints.yaml --auto-oom-retry --resume`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs/post_tonal_without_pcset_constraints.yaml --split test --experiment-name without_pcset_constraints --output C:\Users\nyp\Documents\yinyue2\results\project2_v3_without_pcset_constraints_metrics.json --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --examples-output results/project2_v3_generation_examples.json --export-dir results/eval_musicxml_v3/without_pcset_constraints --checkpoint C:\Users\nyp\Documents\yinyue2\runs\v3\without_pcset_constraints\checkpoint.pt`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.train --config configs/post_tonal_without_serial_constraints.yaml --auto-oom-retry --resume`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs/post_tonal_without_serial_constraints.yaml --split test --experiment-name without_serial_constraints --output C:\Users\nyp\Documents\yinyue2\results\project2_v3_without_serial_constraints_metrics.json --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --examples-output results/project2_v3_generation_examples.json --export-dir results/eval_musicxml_v3/without_serial_constraints --checkpoint C:\Users\nyp\Documents\yinyue2\runs\v3\without_serial_constraints\checkpoint.pt`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.train --config configs/post_tonal_without_rhythm_constraints.yaml --auto-oom-retry --resume`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs/post_tonal_without_rhythm_constraints.yaml --split test --experiment-name without_rhythm_constraints --output C:\Users\nyp\Documents\yinyue2\results\project2_v3_without_rhythm_constraints_metrics.json --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --examples-output results/project2_v3_generation_examples.json --export-dir results/eval_musicxml_v3/without_rhythm_constraints --checkpoint C:\Users\nyp\Documents\yinyue2\runs\v3\without_rhythm_constraints\checkpoint.pt`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.train --config configs/post_tonal_without_gesture_constraints.yaml --auto-oom-retry --resume`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs/post_tonal_without_gesture_constraints.yaml --split test --experiment-name without_gesture_constraints --output C:\Users\nyp\Documents\yinyue2\results\project2_v3_without_gesture_constraints_metrics.json --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --examples-output results/project2_v3_generation_examples.json --export-dir results/eval_musicxml_v3/without_gesture_constraints --checkpoint C:\Users\nyp\Documents\yinyue2\runs\v3\without_gesture_constraints\checkpoint.pt`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.train --config configs/post_tonal_serial_only.yaml --auto-oom-retry --resume`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs/post_tonal_serial_only.yaml --split test --experiment-name serial_only --output C:\Users\nyp\Documents\yinyue2\results\project2_v3_serial_only_metrics.json --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --examples-output results/project2_v3_generation_examples.json --export-dir results/eval_musicxml_v3/serial_only --checkpoint C:\Users\nyp\Documents\yinyue2\runs\v3\serial_only\checkpoint.pt`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.train --config configs/post_tonal_pcset_only.yaml --auto-oom-retry --resume`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run env-check`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m pytest`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run write-split-summary --config configs/post_tonal_main.yaml --output results/project2_v3_full_split_summary.json`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.train --config configs/post_tonal_pcset_only.yaml --auto-oom-retry --resume`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs/post_tonal_pcset_only.yaml --split test --experiment-name pcset_only --output C:\Users\nyp\Documents\yinyue2\results\project2_v3_pcset_only_metrics.json --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --examples-output results/project2_v3_generation_examples.json --export-dir results/eval_musicxml_v3/pcset_only --checkpoint C:\Users\nyp\Documents\yinyue2\runs\v3\pcset_only\checkpoint.pt`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.train --config configs/post_tonal_rhythm_only.yaml --auto-oom-retry --resume`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs/post_tonal_rhythm_only.yaml --split test --experiment-name rhythm_only --output C:\Users\nyp\Documents\yinyue2\results\project2_v3_rhythm_only_metrics.json --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --examples-output results/project2_v3_generation_examples.json --export-dir results/eval_musicxml_v3/rhythm_only --checkpoint C:\Users\nyp\Documents\yinyue2\runs\v3\rhythm_only\checkpoint.pt`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.train --config configs/post_tonal_gesture_only.yaml --auto-oom-retry --resume`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs/post_tonal_gesture_only.yaml --split test --experiment-name gesture_only --output C:\Users\nyp\Documents\yinyue2\results\project2_v3_gesture_only_metrics.json --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --examples-output results/project2_v3_generation_examples.json --export-dir results/eval_musicxml_v3/gesture_only --checkpoint C:\Users\nyp\Documents\yinyue2\runs\v3\gesture_only\checkpoint.pt`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.train --config configs/post_tonal_no_constraints.yaml --auto-oom-retry --resume`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs/post_tonal_no_constraints.yaml --split test --experiment-name no_constraints --output C:\Users\nyp\Documents\yinyue2\results\project2_v3_no_constraints_metrics.json --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --examples-output results/project2_v3_generation_examples.json --export-dir results/eval_musicxml_v3/no_constraints --checkpoint C:\Users\nyp\Documents\yinyue2\runs\v3\no_constraints\checkpoint.pt`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.make_tables --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --main-table paper/tables/project2_v3_main_results.tex --ablation-table paper/tables/project2_v3_ablation_results.tex`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.plot_results --metrics-csv results/project2_v3_metrics.csv --output results/project2_v3_constraint_summary.svg`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.prepare_expert_eval --output-dir expert_eval/project2_v3 --count 20 --examples-json results/project2_v3_generation_examples.json --experiment proposed_constraint_guided_transformer`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.generate --generator transformer --config configs/post_tonal_main.yaml --checkpoint runs/v3/proposed_constraint_guided_transformer/checkpoint.pt --pcset 0,1,4,6 --rhythm_profile pointillistic --gesture fragmented --voices 4 --measures 8 --attempts 4 --num-examples 20 --output-dir results/eval_musicxml_v3/proposed_constraint_guided_transformer`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m pytest`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run write-report --output results/project2_v3_full_run_report.md --metrics results/project2_v3_metrics.csv --constraints results/project2_v3_constraints.csv --examples results/project2_v3_generation_examples.json --split-summary results/project2_v3_full_split_summary.json --expert-dir expert_eval/project2_v3 --run-root runs/v3 --log-path logs/project2_v3_full_run.log --main-table paper/tables/project2_v3_main_results.tex --ablation-table paper/tables/project2_v3_ablation_results.tex`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run write-report --output results/project2_full_run_report.md --metrics results/project2_metrics.csv --constraints results/project2_constraints.csv --examples results/project2_generation_examples.json --split-summary results/project2_full_split_summary.json --expert-dir expert_eval/project2 --run-root runs/v3 --log-path logs/project2_v3_full_run.log --main-table paper/tables/project2_main_results.tex --ablation-table paper/tables/project2_ablation_results.tex`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run env-check`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m pytest`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run write-split-summary --config configs/post_tonal_main.yaml --output results/project2_v3_full_split_summary.json`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.make_tables --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --main-table paper/tables/project2_v3_main_results.tex --ablation-table paper/tables/project2_v3_ablation_results.tex`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.plot_results --metrics-csv results/project2_v3_metrics.csv --output results/project2_v3_constraint_summary.svg`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.prepare_expert_eval --output-dir expert_eval/project2_v3 --count 20 --examples-json results/project2_v3_generation_examples.json --experiment proposed_constraint_guided_transformer`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.generate --generator transformer --config configs/post_tonal_main.yaml --checkpoint runs/v3/proposed_constraint_guided_transformer/checkpoint.pt --pcset 0,1,4,6 --rhythm_profile pointillistic --gesture fragmented --voices 4 --measures 8 --attempts 4 --num-examples 20 --output-dir results/eval_musicxml_v3/proposed_constraint_guided_transformer`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m pytest`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run write-report --output results/project2_v3_full_run_report.md --metrics results/project2_v3_metrics.csv --constraints results/project2_v3_constraints.csv --examples results/project2_v3_generation_examples.json --split-summary results/project2_v3_full_split_summary.json --expert-dir expert_eval/project2_v3 --run-root runs/v3 --log-path logs/project2_v3_full_run.log --main-table paper/tables/project2_v3_main_results.tex --ablation-table paper/tables/project2_v3_ablation_results.tex --incidents results/project2_v3_run_incidents.json`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run write-report --output results/project2_full_run_report.md --metrics results/project2_metrics.csv --constraints results/project2_constraints.csv --examples results/project2_generation_examples.json --split-summary results/project2_full_split_summary.json --expert-dir expert_eval/project2 --run-root runs/v3 --log-path logs/project2_v3_full_run.log --main-table paper/tables/project2_main_results.tex --ablation-table paper/tables/project2_ablation_results.tex --incidents results/project2_full_run_incidents.json`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run env-check`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m pytest`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run write-split-summary --config configs/post_tonal_main.yaml --output results/project2_v3_full_split_summary.json`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.make_tables --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --main-table paper/tables/project2_v3_main_results.tex --ablation-table paper/tables/project2_v3_ablation_results.tex`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.plot_results --metrics-csv results/project2_v3_metrics.csv --output results/project2_v3_constraint_summary.svg`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.prepare_expert_eval --output-dir expert_eval/project2_v3 --count 20 --examples-json results/project2_v3_generation_examples.json --experiment proposed_constraint_guided_transformer`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.generate --generator transformer --config configs/post_tonal_main.yaml --checkpoint runs/v3/proposed_constraint_guided_transformer/checkpoint.pt --pcset 0,1,4,6 --rhythm_profile pointillistic --gesture fragmented --voices 4 --measures 8 --attempts 4 --num-examples 20 --output-dir results/eval_musicxml_v3/proposed_constraint_guided_transformer`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m pytest`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run write-report --output results/project2_v3_full_run_report.md --metrics results/project2_v3_metrics.csv --constraints results/project2_v3_constraints.csv --examples results/project2_v3_generation_examples.json --split-summary results/project2_v3_full_split_summary.json --expert-dir expert_eval/project2_v3 --run-root runs/v3 --log-path logs/project2_v3_full_run.log --main-table paper/tables/project2_v3_main_results.tex --ablation-table paper/tables/project2_v3_ablation_results.tex --incidents results/project2_v3_run_incidents.json`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run write-report --output results/project2_full_run_report.md --metrics results/project2_metrics.csv --constraints results/project2_constraints.csv --examples results/project2_generation_examples.json --split-summary results/project2_full_split_summary.json --expert-dir expert_eval/project2 --run-root runs/v3 --log-path logs/project2_v3_full_run.log --main-table paper/tables/project2_main_results.tex --ablation-table paper/tables/project2_ablation_results.tex --incidents results/project2_full_run_incidents.json`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run env-check`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m pytest`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run write-split-summary --config configs/post_tonal_main.yaml --output results/project2_v3_full_split_summary.json`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.make_tables --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --main-table paper/tables/project2_v3_main_results.tex --ablation-table paper/tables/project2_v3_ablation_results.tex`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.plot_results --metrics-csv results/project2_v3_metrics.csv --output results/project2_v3_constraint_summary.svg`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.prepare_expert_eval --output-dir expert_eval/project2_v3 --count 20 --examples-json results/project2_v3_generation_examples.json --experiment proposed_constraint_guided_transformer`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.generate --generator transformer --config configs/post_tonal_main.yaml --checkpoint runs/v3/proposed_constraint_guided_transformer/checkpoint.pt --pcset 0,1,4,6 --rhythm_profile pointillistic --gesture fragmented --voices 4 --measures 8 --attempts 4 --num-examples 20 --output-dir results/eval_musicxml_v3/proposed_constraint_guided_transformer`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m pytest`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run write-report --output results/project2_v3_full_run_report.md --metrics results/project2_v3_metrics.csv --constraints results/project2_v3_constraints.csv --examples results/project2_v3_generation_examples.json --split-summary results/project2_v3_full_split_summary.json --expert-dir expert_eval/project2_v3 --run-root runs/v3 --log-path logs/project2_v3_full_run.log --main-table paper/tables/project2_v3_main_results.tex --ablation-table paper/tables/project2_v3_ablation_results.tex --incidents results/project2_v3_run_incidents.json`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run promote-generation-examples --source results/project2_v3_generation_examples.json --output results/project2_generation_examples.json --source-root results/eval_musicxml_v3 --destination-root results/eval_musicxml`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.prepare_expert_eval --output-dir expert_eval/project2 --count 20 --examples-json results/project2_generation_examples.json --experiment proposed_constraint_guided_transformer`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run write-report --output results/project2_full_run_report.md --metrics results/project2_metrics.csv --constraints results/project2_constraints.csv --examples results/project2_generation_examples.json --split-summary results/project2_full_split_summary.json --expert-dir expert_eval/project2 --run-root runs/v3 --log-path logs/project2_v3_full_run.log --main-table paper/tables/project2_main_results.tex --ablation-table paper/tables/project2_ablation_results.tex --incidents results/project2_full_run_incidents.json`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run env-check`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m pytest`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run write-split-summary --config configs/post_tonal_main.yaml --output results/project2_v3_full_split_summary.json`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs/post_tonal_rule_baseline.yaml --split test --experiment-name rule_baseline --output C:\Users\nyp\Documents\yinyue2\results\project2_v3_rule_baseline_metrics.json --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --examples-output results/project2_v3_generation_examples.json --export-dir results/eval_musicxml_v3/rule_baseline`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.make_tables --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --main-table paper/tables/project2_v3_main_results.tex --ablation-table paper/tables/project2_v3_ablation_results.tex`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.plot_results --metrics-csv results/project2_v3_metrics.csv --output results/project2_v3_constraint_summary.svg`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.prepare_expert_eval --output-dir expert_eval/project2_v3 --count 20 --examples-json results/project2_v3_generation_examples.json --experiment proposed_constraint_guided_transformer`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.generate --generator transformer --config configs/post_tonal_main.yaml --checkpoint runs/v3/proposed_constraint_guided_transformer/checkpoint.pt --pcset 0,1,4,6 --rhythm_profile pointillistic --gesture fragmented --voices 4 --measures 8 --attempts 4 --num-examples 20 --output-dir results/eval_musicxml_v3/proposed_constraint_guided_transformer`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m pytest`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run write-report --output results/project2_v3_full_run_report.md --metrics results/project2_v3_metrics.csv --constraints results/project2_v3_constraints.csv --examples results/project2_v3_generation_examples.json --split-summary results/project2_v3_full_split_summary.json --expert-dir expert_eval/project2_v3 --run-root runs/v3 --log-path logs/project2_v3_full_run.log --main-table paper/tables/project2_v3_main_results.tex --ablation-table paper/tables/project2_v3_ablation_results.tex --incidents results/project2_v3_run_incidents.json`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run promote-generation-examples --source results/project2_v3_generation_examples.json --output results/project2_generation_examples.json --source-root results/eval_musicxml_v3 --destination-root results/eval_musicxml`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.prepare_expert_eval --output-dir expert_eval/project2 --count 20 --examples-json results/project2_generation_examples.json --experiment proposed_constraint_guided_transformer`
- `C:\Users\nyp\Documents\yinyue2\.venv311\Scripts\python.exe -m post_tonal.full_run write-report --output results/project2_full_run_report.md --metrics results/project2_metrics.csv --constraints results/project2_constraints.csv --examples results/project2_generation_examples.json --split-summary results/project2_full_split_summary.json --expert-dir expert_eval/project2 --run-root runs/v3 --log-path logs/project2_v3_full_run.log --main-table paper/tables/project2_main_results.tex --ablation-table paper/tables/project2_ablation_results.tex --incidents results/project2_full_run_incidents.json`

## Environment Information
- Python: 3.11.9
- PyTorch: 2.5.1+cu121
- CUDA available: True
- CUDA device: NVIDIA GeForce RTX 4060 Ti
- Peak process RAM across completed training runs: 4.637 GiB
- Peak allocated CUDA memory across completed training runs: 0.667 GiB

## CUDA Check Output
- cuda_available: True
- cuda_device_count: 1
- cuda_device_name_0: NVIDIA GeForce RTX 4060 Ti

## Corpus Split Counts
- Train: 20000
- Validation: 2000
- Test: 2000
- Smoke: False
- Corpus format: post_tonal_synthetic_v3_windowed
- Sequence strategy: coverage_cycle
- Raw score-body tokens discarded by training: 0

## Experiment Configs Completed
- rule_baseline
- vanilla_transformer
- proposed_constraint_guided_transformer
- transformer_no_constraints
- without_pcset_constraints
- without_serial_constraints
- without_rhythm_constraints
- without_gesture_constraints
- serial_only
- pcset_only
- rhythm_only
- gesture_only
- no_constraints

## Neural Checkpoints Completed
- vanilla_transformer
- proposed_constraint_guided_transformer
- transformer_no_constraints
- without_pcset_constraints
- without_serial_constraints
- without_rhythm_constraints
- without_gesture_constraints
- serial_only
- pcset_only
- rhythm_only
- gesture_only
- no_constraints

## Neural Checkpoint Details
- vanilla_transformer: epochs=60, best_epoch=60, stop_reason=max_epochs, checkpoint=runs/v3/vanilla_transformer/checkpoint.pt, sha256=35ef1047b1c0eda553586d48c3402eb6f35cf38603032292e05bdc9cf40ff61d
- proposed_constraint_guided_transformer: epochs=60, best_epoch=60, stop_reason=max_epochs, checkpoint=runs/v3/proposed_constraint_guided_transformer/checkpoint.pt, sha256=35ef1047b1c0eda553586d48c3402eb6f35cf38603032292e05bdc9cf40ff61d
- transformer_no_constraints: epochs=60, best_epoch=56, stop_reason=max_epochs, checkpoint=runs/v3/transformer_no_constraints/checkpoint.pt, sha256=53c52dc0204e8e59ebdcb88afcdf77e94f32b119b58d2fed8740264ce52bc90c
- without_pcset_constraints: epochs=60, best_epoch=60, stop_reason=max_epochs, checkpoint=runs/v3/without_pcset_constraints/checkpoint.pt, sha256=8d9fd81413b29cb326d5ce3f8018eca19e7f098ed72d09b74ff852e8292e2df5
- without_serial_constraints: epochs=60, best_epoch=60, stop_reason=max_epochs, checkpoint=runs/v3/without_serial_constraints/checkpoint.pt, sha256=04b9dfcdcd794f8728518ae85966806886c42c274d200cd87440cfae365b03c0
- without_rhythm_constraints: epochs=60, best_epoch=60, stop_reason=max_epochs, checkpoint=runs/v3/without_rhythm_constraints/checkpoint.pt, sha256=f6ef9d698d263741ee875610505eb9418e4d6a2e46aa3a693d88db3d17320d45
- without_gesture_constraints: epochs=60, best_epoch=58, stop_reason=max_epochs, checkpoint=runs/v3/without_gesture_constraints/checkpoint.pt, sha256=1f4a8be715be43f518acfde6a2534aec2bc6f6a407e46c926866d5f1b2834daa
- serial_only: epochs=60, best_epoch=60, stop_reason=max_epochs, checkpoint=runs/v3/serial_only/checkpoint.pt, sha256=491ca6f39235dd96e3b930e27e0708fbbdcea9957b5e0c262eac73a11954c9d0
- pcset_only: epochs=60, best_epoch=59, stop_reason=max_epochs, checkpoint=runs/v3/pcset_only/checkpoint.pt, sha256=d9b8a9f20ba09dfc383af7df2f6ddfe22a7e8d548b1d833f76cd85d88a045504
- rhythm_only: epochs=60, best_epoch=60, stop_reason=max_epochs, checkpoint=runs/v3/rhythm_only/checkpoint.pt, sha256=ff3d51dcdaa20c70046cdc0a39c5a3a85b01f303e05d07aad84fd5099604ba06
- gesture_only: epochs=60, best_epoch=60, stop_reason=max_epochs, checkpoint=runs/v3/gesture_only/checkpoint.pt, sha256=5325ebfa7addcdf6b65a43ff5019ea1d8a9680f7c589be174c144f61e41acab4
- no_constraints: epochs=60, best_epoch=60, stop_reason=max_epochs, checkpoint=runs/v3/no_constraints/checkpoint.pt, sha256=e0bb5cac1ce87d061872e05c76bd9190f83abf6c90ae0c1c671685ab12649a2d

## Pending Evaluation Rows
- None.

## Missing or Incomplete Neural Checkpoints
- None.

## Failed or Retried Stages
- pcset_only training startup [recovered]: Python exited with -1073740791 (0xC0000409). Windows Application Error event 1000 identified nvcuda64.dll 32.0.15.9186 as the faulting module. Recovery: A CUDA fp16 4096-by-4096 matrix multiplication health check succeeded, then the full wrapper was restarted with -Resume. pcset_only subsequently completed 60 epochs and its 2000-sample test evaluation. Evidence: Windows report ID ce14a5f2-af73-41cf-afe2-2fb4d61f681c; logs/project2_v3_wrapper_20260716_223610.stderr.log; runs/v3/pcset_only/train_summary.json

## OOM Adjustments
- None recorded in completed training summaries.

## Final Metrics File Paths
- results/project2_metrics.csv
- results/project2_constraints.csv
- results/project2_generation_examples.json
- results/project2_full_split_summary.json

## Generated MusicXML Examples Path
- expert_eval/project2/musicxml/ (20/20 structurally score-partwise; 20/20 requested-span adherent; 20/20 requested-voice adherent)

## Paper Tables Path
- paper/tables/project2_main_results.tex (present)
- paper/tables/project2_ablation_results.tex (present)

## Remaining TODOs
- Add blind expert ratings after human evaluation.
- Add independent, legally supplied MusicXML validation examples when available.
- Complete author metadata, declarations, and live journal-portal checks.
