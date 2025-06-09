"""
Script to log and register LLaMA2-7B model to MLflow for HP AI Studio deployment.
"""
import os
import mlflow
import mlflow.pyfunc
import argparse
import pandas as pd

class Llama2Model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):  # pylint: disable=unused-argument
        model_path = context.artifacts.get("model_path")
        from langchain_community.llms import LlamaCpp

        # Load LLaMA2 model via llama.cpp bindings
        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=int(os.environ.get("LLAMA_N_GPU_LAYERS", -1)),
            n_batch=int(os.environ.get("LLAMA_N_BATCH", 512)),
            n_ctx=int(os.environ.get("LLAMA_N_CTX", 4096)),
            max_tokens=int(os.environ.get("LLAMA_MAX_TOKENS", 1024)),
            f16_kv=bool(int(os.environ.get("LLAMA_F16_KV", 1))),
            verbose=False,
            stop=[],
            temperature=float(os.environ.get("LLAMA_TEMPERATURE", 0.7)),
        )

    def predict(self, context, model_input):  # type: ignore
        # model_input: pandas DataFrame with column 'prompt'
        prompt = model_input.get("prompt").iloc[0]
        result = self.llm(prompt)
        if isinstance(result, dict):
            return pd.Series([result.get("text", str(result))])
        return pd.Series([str(result)])

def main():
    parser = argparse.ArgumentParser(
        description="Log and register LLaMA2 model to MLflow registry"
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to local gguf model file (e.g., ggml-model-f16.Q5_K_M.gguf)"
    )
    parser.add_argument(
        "--model-name", default="llama2-7b",
        help="Name to register the model under in MLflow registry"
    )
    parser.add_argument(
        "--artifact-path", default="model",
        help="Artifact path within the MLflow run"
    )
    parser.add_argument(
        "--register", action="store_true",
        help="Automatically register model after logging"
    )
    args = parser.parse_args()

    # Start MLflow run
    with mlflow.start_run() as run:
        artifacts = {"model_path": args.model_path}
        mlflow.pyfunc.log_model(
            artifact_path=args.artifact_path,
            python_model=Llama2Model(),
            artifacts=artifacts,
            registered_model_name=(args.model_name if args.register else None)
        )
        print(
            f"Model logged in run {run.info.run_id}, artifacts under '{args.artifact_path}'."
        )
        if args.register:
            print(
                f"Model registered in MLflow registry as '{args.model_name}'."
            )
        else:
            print(
                "To register the model, run with --register or use MLflow UI/CLI:"
            )
            print(
                f"  mlflow register-model -m runs:/{run.info.run_id}/{args.artifact_path} -n {args.model_name}"
            )

if __name__ == "__main__":
    main()