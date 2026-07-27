from __future__ import annotations

import csv
from pathlib import Path

import torch

from local_rsl_rl.runners import OnPolicyRunner


def _get_pyplot(live_plot: bool = False):
    import matplotlib

    if live_plot:
        backend = matplotlib.get_backend().lower()
        if "agg" in backend:
            for candidate in ("TkAgg", "Qt5Agg", "QtAgg"):
                try:
                    matplotlib.use(candidate, force=True)
                    break
                except Exception:
                    continue
    import matplotlib.pyplot as plt

    return plt


class PlayRouterWeightLogger:
    """Record per-task MoE router weights during play and plot them live or on exit."""

    def __init__(
        self,
        task_names: tuple[str, ...] | list[str],
        expert_names: tuple[str, ...] | list[str],
        output_dir: str | Path,
        sample_interval: int = 1,
        live_plot: bool = False,
        routing_mode: str = "soft",
        inference_mode: str = "teacher",
        max_points: int = 2000,
    ):
        if sample_interval < 1:
            raise ValueError("sample_interval must be >= 1")

        self.task_names = list(task_names)
        self.expert_names = list(expert_names)
        self.output_dir = Path(output_dir)
        self.sample_interval = sample_interval
        self.live_plot = live_plot
        self.routing_mode = routing_mode
        self.inference_mode = inference_mode
        self.max_points = max_points

        self._steps: dict[str, list[int]] = {task_name: [] for task_name in self.task_names}
        self._weights: dict[str, dict[str, list[float]]] = {
            task_name: {expert_name: [] for expert_name in self.expert_names}
            for task_name in self.task_names
        }

        self._plt = None
        self._fig = None
        self._axes: dict[str, object] = {}
        self._lines: dict[str, dict[str, object]] = {}
        if self.live_plot:
            self._setup_live_plot()

    def _setup_live_plot(self) -> None:
        self._plt = _get_pyplot(live_plot=True)
        num_tasks = len(self.task_names)
        self._fig, axes = self._plt.subplots(num_tasks, 1, figsize=(10, 3.2 * num_tasks), sharex=True)
        if num_tasks == 1:
            axes = [axes]

        for axis, task_name in zip(axes, self.task_names):
            self._axes[task_name] = axis
            self._lines[task_name] = {}
            for expert_name in self.expert_names:
                line, = axis.plot([], [], linewidth=1.8, label=expert_name)
                self._lines[task_name][expert_name] = line
            axis.set_title(task_name)
            axis.set_ylim(0.0, 1.0)
            axis.set_xlim(0.0, 10.0)
            axis.grid(True, alpha=0.3)
            axis.legend(loc="best", fontsize=8)

        axes[-1].set_xlabel("Play step")
        self._fig.suptitle(
            f"MoE Router Weights Live ({self.routing_mode}, {self.inference_mode})"
        )
        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        self._plt.pause(0.001)

    def maybe_record(
        self,
        step: int,
        router_weights: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> None:
        if step % self.sample_interval != 0:
            return

        router_entries = OnPolicyRunner.aggregate_router_weights_by_task(
            router_weights,
            task_ids,
            self.task_names,
            self.expert_names,
        )
        updated = False
        for task_name in self.task_names:
            if not any(key.startswith(f"{task_name}/") for key in router_entries):
                continue
            self._steps[task_name].append(step)
            for expert_name in self.expert_names:
                self._weights[task_name][expert_name].append(
                    router_entries.get(f"{task_name}/{expert_name}", 0.0)
                )
            self._trim_history(task_name)
            updated = True

        if updated and self.live_plot:
            self.refresh_live_plot()

    def _trim_history(self, task_name: str) -> None:
        overflow = len(self._steps[task_name]) - self.max_points
        if overflow <= 0:
            return
        self._steps[task_name] = self._steps[task_name][overflow:]
        for expert_name in self.expert_names:
            self._weights[task_name][expert_name] = self._weights[task_name][expert_name][overflow:]

    def refresh_live_plot(self) -> None:
        if not self.live_plot or self._fig is None:
            return

        max_step = 0
        for task_name in self.task_names:
            steps = self._steps[task_name]
            if not steps:
                continue
            max_step = max(max_step, steps[-1])
            axis = self._axes[task_name]
            for expert_name in self.expert_names:
                line = self._lines[task_name][expert_name]
                line.set_data(steps, self._weights[task_name][expert_name])
            axis.relim()
            axis.autoscale_view(scalex=True, scaley=False)
            axis.set_ylim(0.0, 1.0)

        if max_step > 0:
            for axis in self._axes.values():
                axis.set_xlim(0.0, max(max_step, 10))

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._plt.pause(0.001)

    def has_data(self) -> bool:
        return any(len(steps) > 0 for steps in self._steps.values())

    def close(self) -> None:
        if self._fig is not None and self._plt is not None:
            self._plt.close(self._fig)
            self._fig = None

    def save(self, routing_mode: str | None = None, inference_mode: str | None = None) -> list[Path]:
        if not self.has_data():
            return []

        routing_mode = self.routing_mode if routing_mode is None else routing_mode
        inference_mode = self.inference_mode if inference_mode is None else inference_mode
        plt = self._plt if self._plt is not None else _get_pyplot(live_plot=False)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []

        saved_paths.extend(self._save_csv(routing_mode, inference_mode))
        saved_paths.extend(self._save_task_plots(routing_mode, inference_mode, plt))
        saved_paths.append(self._save_overview_plot(routing_mode, inference_mode, plt))
        return saved_paths

    def _save_csv(self, routing_mode: str, inference_mode: str) -> list[Path]:
        csv_path = self.output_dir / f"router_weights_{routing_mode}_{inference_mode}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["task", "step", *self.expert_names])
            for task_name in self.task_names:
                steps = self._steps[task_name]
                if not steps:
                    continue
                for idx, step in enumerate(steps):
                    row = [task_name, step]
                    row.extend(self._weights[task_name][expert_name][idx] for expert_name in self.expert_names)
                    writer.writerow(row)
        return [csv_path]

    def _save_task_plots(self, routing_mode: str, inference_mode: str, plt) -> list[Path]:
        saved_paths: list[Path] = []
        for task_name in self.task_names:
            steps = self._steps[task_name]
            if not steps:
                continue

            fig, axis = plt.subplots(figsize=(10, 5))
            for expert_name in self.expert_names:
                axis.plot(
                    steps,
                    self._weights[task_name][expert_name],
                    linewidth=2.0,
                    label=expert_name,
                )

            axis.set_title(f"MoE Router Weights - {task_name}")
            axis.set_xlabel("Play step")
            axis.set_ylabel("Router weight")
            axis.set_ylim(0.0, 1.0)
            axis.grid(True, alpha=0.3)
            axis.legend(loc="best")

            fig.tight_layout()
            plot_path = self.output_dir / f"router_weights_{task_name}_{routing_mode}_{inference_mode}.png"
            fig.savefig(plot_path, dpi=160)
            plt.close(fig)
            saved_paths.append(plot_path)
        return saved_paths

    def _save_overview_plot(self, routing_mode: str, inference_mode: str, plt) -> Path:
        active_tasks = [task_name for task_name in self.task_names if self._steps[task_name]]
        num_tasks = len(active_tasks)
        fig, axes = plt.subplots(num_tasks, 1, figsize=(10, 3.5 * num_tasks), sharex=True)
        if num_tasks == 1:
            axes = [axes]

        for axis, task_name in zip(axes, active_tasks):
            for expert_name in self.expert_names:
                axis.plot(
                    self._steps[task_name],
                    self._weights[task_name][expert_name],
                    linewidth=1.8,
                    label=expert_name,
                )
            axis.set_title(task_name)
            axis.set_ylim(0.0, 1.0)
            axis.grid(True, alpha=0.3)
            axis.legend(loc="best", fontsize=8)

        axes[-1].set_xlabel("Play step")
        fig.suptitle(f"MoE Router Weights ({routing_mode}, {inference_mode})")
        fig.tight_layout()

        plot_path = self.output_dir / f"router_weights_overview_{routing_mode}_{inference_mode}.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        return plot_path
