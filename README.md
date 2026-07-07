# Neural Networks: Why They Do Not Always Overfit

A standalone [marimo](https://marimo.io) teaching app for ML4Prediction.

The app uses one XOR dataset to make a single question concrete: if neural
networks can fit many parameters, why do they not always overfit? Students
compare decision boundaries, inspect the number of fitted weights and biases,
watch training and validation loss diverge, and then regularize the same model
with dropout plus L1/L2 penalties.

It runs entirely in the browser as a static WebAssembly bundle, with no server,
external data, `torch`, TensorFlow, or scikit-learn.

## Live demo

**<https://javier.science/marimo_nn_overfitting/>**

The bundle is rebuilt automatically by GitHub Actions on every push to `main`
(see [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)).

## Running locally

```bash
uv sync
sh run.sh
```

## Rebuilding the WASM bundle locally

```bash
bash export_wasm.sh
```

Preview the generated `build/` folder with a local static server.

## License

MIT.
