#!/usr/bin/env python3
"""Run Lami TFLite model with the GPU delegate via tflite-runtime."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Lami TFLite model using the GPU delegate via LiteRT"
    )
    parser.add_argument(
        "model",
        help="Path to the .tflite model file"
    )
    args = parser.parse_args()

    # Imports are done after argument parsing so --help works without them
    import numpy as np
    from tflite_runtime.interpreter import Interpreter, load_delegate

    delegate = load_delegate("libtensorflowlite_gpu_delegate.so")
    interpreter = Interpreter(
        model_path=args.model,
        experimental_delegates=[delegate],
    )
    interpreter.allocate_tensors()

    # Populate inputs with zeros
    for detail in interpreter.get_input_details():
        data = np.zeros(detail["shape"], dtype=detail["dtype"])
        interpreter.set_tensor(detail["index"], data)

    interpreter.invoke()

    for i, detail in enumerate(interpreter.get_output_details()):
        output = interpreter.get_tensor(detail["index"])
        print(f"Output {i}: {output}")


if __name__ == "__main__":
    main()
