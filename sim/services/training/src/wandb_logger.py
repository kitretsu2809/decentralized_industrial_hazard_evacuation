from typing import Dict

class TrainingLogger:
    def __init__(self, project: str = 'lbp-evacuation', enabled: bool = True):
        self.enabled = enabled
        self.project = project
        if self.enabled:
            try:
                import wandb
                wandb.init(project=self.project)
                self.wandb = wandb
            except ImportError:
                print("WandB not installed. Falling back to console logging.")
                self.wandb = None
        else:
            self.wandb = None

    def log_episode(self, episode: int, metrics: Dict):
        if self.wandb:
            self.wandb.log(metrics, step=episode)
        else:
            print(f"Episode {episode}: {metrics}")

    def log_evaluation(self, scenario: str, metrics: Dict):
        if self.wandb:
            eval_metrics = {f"eval_{scenario}/{k}": v for k, v in metrics.items()}
            self.wandb.log(eval_metrics)
        else:
            print(f"Evaluation {scenario}: {metrics}")

    def log_curriculum_stage(self, stage: Dict):
        if self.wandb:
            self.wandb.log({"curriculum_stage": stage})
        else:
            print(f"Curriculum Stage: {stage}")

    def save_model_artifact(self, model_path: str):
        if self.wandb:
            artifact = self.wandb.Artifact('model', type='model')
            artifact.add_file(model_path)
            self.wandb.log_artifact(artifact)
        else:
            print(f"Model saved at {model_path}")

    def finish(self):
        if self.wandb:
            self.wandb.finish()
