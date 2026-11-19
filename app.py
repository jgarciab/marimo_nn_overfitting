import marimo

__generated_with = "0.23.0"
app = marimo.App(
    width="full",
    app_title="Neural Networks: Why They Do Not Always Overfit",
)


@app.cell
def imports():
    import marimo as mo

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#333333",
            "text.color": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "figure.dpi": 130,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": "#cccccc",
        }
    )
    return matplotlib, mo, np, plt


@app.cell
def theme(mo):
    uu_blue = "#0b789d"
    class_purple = "#5e548e"
    class_gold = "#c9a227"
    train_color = "#333333"
    validation_color = "#0b789d"
    text_gray = "#333333"

    css = mo.Html(
        f"""
        <style>
        :root {{
          --ml4p-accent: {uu_blue};
          --ml4p-text: {text_gray};
        }}
        .ml4p-callout {{
          border-left: 4px solid var(--ml4p-accent);
          padding: 0.65rem 0.85rem;
          background: color-mix(in srgb, var(--ml4p-accent) 8%, white);
          margin: 0.4rem 0 0.8rem;
        }}
        .ml4p-small {{
          color: #666666;
          font-size: 0.9rem;
        }}
        .ml4p-param-box {{
          border: 1px solid #dddddd;
          border-radius: 6px;
          padding: 0.75rem 0.85rem;
          background: #fbfbfb;
          margin: 0.25rem 0 0.9rem;
        }}
        .ml4p-param-number {{
          color: var(--ml4p-accent);
          font-size: 1.65rem;
          font-weight: 700;
          line-height: 1.1;
        }}
        </style>
        """
    )
    return class_gold, class_purple, css, train_color, validation_color


@app.cell
def capacity_controls(mo):
    capacity_n = mo.ui.slider(40, 200, step=20, value=100, label="Sample size")
    capacity_noise = mo.ui.slider(0.0, 0.5, step=0.05, value=0.2, label="Label noise")
    capacity_layers = mo.ui.slider(1, 6, step=1, value=4, label="Hidden layers")
    capacity_units = mo.ui.slider(2, 128, step=2, value=64, label="Units per layer")
    capacity_epochs = mo.ui.slider(20, 500, step=20, value=300, label="Epochs")
    capacity_activation = mo.ui.dropdown(
        options=["ReLU", "tanh", "linear"],
        value="ReLU",
        label="Activation",
    )
    dropout_rate = mo.ui.slider(0.0, 0.6, step=0.05, value=0.0, label="Dropout rate")
    l2_strength = mo.ui.slider(0.0, 0.05, step=0.001, value=0.0, label="L2 penalty")
    return (
        capacity_activation,
        capacity_epochs,
        capacity_layers,
        capacity_n,
        capacity_noise,
        capacity_units,
        dropout_rate,
        l2_strength,
    )


@app.cell
def sidebar(
    capacity_activation,
    capacity_epochs,
    capacity_layers,
    capacity_n,
    capacity_noise,
    capacity_units,
    dropout_rate,
    l2_strength,
    mo,
):
    mo.sidebar(
        [
            mo.md("### Controls"),
            mo.md("**XOR data**"),
            capacity_n,
            capacity_noise,
            mo.md("**Network**"),
            capacity_layers,
            capacity_units,
            capacity_epochs,
            capacity_activation,
            mo.md("**Regularization**"),
            dropout_rate,
            l2_strength,
            mo.md(
                """
                <span class="ml4p-small">The second fit keeps the same data,
                architecture, initial weights, and epoch count; only
                regularization changes.</span>
                """
            ),
        ]
    )
    return


@app.cell
def title(css, mo):
    mo.vstack(
        [
            css,
            mo.md(
                """
                # Neural Networks: Why They Do Not Always Overfit

                *Javier Garcia-Bernardo -- ODISSEI Social Data Science team
                (SoDa) & Department of Methodology and Statistics, Utrecht
                University*

                **Motivating question:** neural networks can have many, many
                fitted parameters. Why do they not always overfit?
                """
            ),
        ]
    )
    return


@app.cell
def data_helpers(np):
    def make_xor_data(n: int, noise: float, seed: int):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-2.2, 2.2, size=(n, 2))
        y = (x[:, 0] * x[:, 1] > 0).astype(int)
        flip = rng.random(n) < noise
        y[flip] = 1 - y[flip]
        return x.astype(float), y.astype(int)

    def train_validation_split(x, y, validation_fraction: float, seed: int):
        rng = np.random.default_rng(seed + 991)
        order = rng.permutation(len(y))
        cut = int(round((1 - validation_fraction) * len(y)))
        train_idx = order[:cut]
        validation_idx = order[cut:]
        return x[train_idx], y[train_idx], x[validation_idx], y[validation_idx]

    return make_xor_data, train_validation_split


@app.cell
def mlp_helpers(np):
    def sigmoid(z):
        z = np.clip(z, -40, 40)
        return 1.0 / (1.0 + np.exp(-z))

    def activation_forward(z, activation: str):
        if activation == "ReLU":
            return np.maximum(z, 0.0)
        if activation == "tanh":
            return np.tanh(z)
        return z

    def activation_backward(z, activation: str):
        if activation == "ReLU":
            return (z > 0).astype(float)
        if activation == "tanh":
            return 1.0 - np.tanh(z) ** 2
        return np.ones_like(z)

    def binary_cross_entropy(y, p):
        y = y.reshape(-1, 1)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def architecture_dims(n_features: int, hidden_layers: int, hidden_units: int):
        return [n_features] + [hidden_units] * hidden_layers + [1]

    def compute_param_count(n_features: int, hidden_layers: int, hidden_units: int):
        dims = architecture_dims(n_features, hidden_layers, hidden_units)
        return int(
            sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
        )

    def parameter_breakdown(n_features: int, hidden_layers: int, hidden_units: int):
        rows = []
        first_params = n_features * hidden_units + hidden_units
        rows.append(
            [
                "input -> hidden 1",
                f"`X_(n x {n_features}) @ W1_({n_features} x {hidden_units}) + b1_({hidden_units})`",
                f"`{n_features} * {hidden_units} + {hidden_units} = {first_params}`",
            ]
        )
        for layer in range(2, hidden_layers + 1):
            params = hidden_units * hidden_units + hidden_units
            rows.append(
                [
                    f"hidden {layer - 1} -> hidden {layer}",
                    f"`H{layer - 1}_(n x {hidden_units}) @ W{layer}_({hidden_units} x {hidden_units}) + b{layer}_({hidden_units})`",
                    f"`{hidden_units} * {hidden_units} + {hidden_units} = {params}`",
                ]
            )
        output_params = hidden_units * 1 + 1
        rows.append(
            [
                f"hidden {hidden_layers} -> output",
                f"`H{hidden_layers}_(n x {hidden_units}) @ Wout_({hidden_units} x 1) + bout_(1)`",
                f"`{hidden_units} * 1 + 1 = {output_params}`",
            ]
        )
        total = compute_param_count(n_features, hidden_layers, hidden_units)
        table_rows = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
        return "\n".join(
            [
                "**Where do the fitted parameters come from?**",
                "",
                "Each layer fits a weight matrix plus a bias vector. For this network:",
                "",
                "| Layer | Matrix multiplication | Parameters |",
                "|---|---|---:|",
                table_rows,
                "",
                f"**Total:** `{total:,}` fitted parameters.",
            ]
        )

    def initialize_model(n_features, hidden_layers, hidden_units, activation, seed):
        rng = np.random.default_rng(seed)
        dims = architecture_dims(n_features, hidden_layers, hidden_units)
        weights = []
        biases = []
        for i in range(len(dims) - 1):
            if activation == "ReLU" and i < len(dims) - 2:
                scale = np.sqrt(2.0 / dims[i])
            else:
                scale = np.sqrt(1.0 / dims[i])
            weights.append(rng.normal(0, scale, size=(dims[i], dims[i + 1])))
            biases.append(np.zeros((1, dims[i + 1])))
        return {"weights": weights, "biases": biases, "activation": activation}

    def forward_pass(model, x, training=False, dropout_rate=0.0, rng=None):
        a = x
        caches = []
        last = len(model["weights"]) - 1
        for i, (w, b) in enumerate(zip(model["weights"], model["biases"])):
            z = a @ w + b
            if i == last:
                a_next = sigmoid(z)
                mask = None
            else:
                a_next = activation_forward(z, model["activation"])
                mask = None
                if training and dropout_rate > 0:
                    keep = 1.0 - dropout_rate
                    mask = (rng.random(a_next.shape) < keep) / keep
                    a_next = a_next * mask
            caches.append((a, z, mask))
            a = a_next
        return a, caches

    def predict_proba(model, x):
        p, _ = forward_pass(model, x, training=False)
        return p.ravel()

    def accuracy(model, x, y):
        pred = (predict_proba(model, x) >= 0.5).astype(int)
        return float(np.mean(pred == y))

    def train_mlp(
        x_train,
        y_train,
        x_validation,
        y_validation,
        hidden_layers=2,
        hidden_units=16,
        activation="ReLU",
        learning_rate=0.035,
        epochs=200,
        batch_size=32,
        dropout_rate=0.0,
        l2=0.0,
        seed=1,
    ):
        rng = np.random.default_rng(seed + 41)
        model = initialize_model(
            x_train.shape[1],
            hidden_layers,
            hidden_units,
            activation,
            seed,
        )
        y_col = y_train.reshape(-1, 1)
        train_loss = []
        validation_loss = []
        validation_accuracy = []

        for epoch in range(epochs):
            order = rng.permutation(len(y_train))
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                xb = x_train[idx]
                yb = y_col[idx]
                p, caches = forward_pass(
                    model,
                    xb,
                    training=True,
                    dropout_rate=dropout_rate,
                    rng=rng,
                )
                dz = (p - yb) / len(yb)
                grad_w = [None] * len(model["weights"])
                grad_b = [None] * len(model["biases"])
                da = None
                last = len(model["weights"]) - 1
                for i in range(last, -1, -1):
                    a_prev, z, mask = caches[i]
                    if i == last:
                        local_dz = dz
                    else:
                        local_da = da
                        if mask is not None:
                            local_da = local_da * mask
                        local_dz = local_da * activation_backward(z, model["activation"])
                    grad_w[i] = a_prev.T @ local_dz
                    grad_w[i] += l2 * model["weights"][i]
                    grad_b[i] = np.sum(local_dz, axis=0, keepdims=True)
                    if i > 0:
                        da = local_dz @ model["weights"][i].T
                for i in range(len(model["weights"])):
                    model["weights"][i] -= learning_rate * np.clip(grad_w[i], -5.0, 5.0)
                    model["biases"][i] -= learning_rate * np.clip(grad_b[i], -5.0, 5.0)

            p_train = forward_pass(model, x_train, training=False)[0]
            p_validation = forward_pass(model, x_validation, training=False)[0]
            train_loss.append(binary_cross_entropy(y_train, p_train))
            validation_loss.append(binary_cross_entropy(y_validation, p_validation))
            validation_accuracy.append(
                float(np.mean((p_validation.ravel() >= 0.5) == y_validation))
            )

        final_train = forward_pass(model, x_train, training=False)[0]
        final_validation = forward_pass(model, x_validation, training=False)[0]
        train_loss = np.array(train_loss)
        validation_loss = np.array(validation_loss)
        validation_accuracy = np.array(validation_accuracy)
        best_idx = int(np.argmin(validation_loss))
        return {
            "model": model,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_accuracy_history": validation_accuracy,
            "final_train_loss": binary_cross_entropy(y_train, final_train),
            "final_validation_loss": binary_cross_entropy(y_validation, final_validation),
            "best_epoch": best_idx + 1,
            "best_validation_loss": float(validation_loss[best_idx]),
            "best_validation_accuracy": float(validation_accuracy[best_idx]),
            "train_accuracy": accuracy(model, x_train, y_train),
            "validation_accuracy": accuracy(model, x_validation, y_validation),
        }

    return compute_param_count, parameter_breakdown, predict_proba, train_mlp


@app.cell
def plotting_helpers(np, plt):
    from matplotlib.colors import LinearSegmentedColormap

    def remove_outer_box(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#bbbbbb")
        ax.spines["bottom"].set_color("#bbbbbb")

    def class_colors(y, class_purple, class_gold):
        colors = np.array([class_purple, class_gold])
        return colors[y.astype(int)]

    def percent(value):
        return f"{100 * value:.0f}%"

    def true_xor_class(x):
        return (x[:, 0] * x[:, 1] > 0).astype(int)

    def binary_log_loss(y, p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def bayes_metrics(x, y, noise):
        true_class = true_xor_class(x)
        p = np.where(true_class == 1, 1 - noise, noise)
        return {
            "accuracy": float(np.mean(true_class == y)),
            "loss": binary_log_loss(y, p),
        }

    def draw_boundary(
        ax,
        result,
        x_train,
        y_train,
        x_validation,
        y_validation,
        predict_proba,
        title,
        class_purple,
        class_gold,
        train_color,
        validation_color,
    ):
        boundary_cmap = LinearSegmentedColormap.from_list(
            "xor_purple_gold",
            [class_purple, "#f7f7f7", class_gold],
        )
        grid_x, grid_y = np.meshgrid(
            np.linspace(-2.7, 2.7, 76),
            np.linspace(-2.7, 2.7, 76),
        )
        grid = np.c_[grid_x.ravel(), grid_y.ravel()]
        p = predict_proba(result["model"], grid).reshape(grid_x.shape)
        ax.contourf(
            grid_x,
            grid_y,
            p,
            levels=np.linspace(0, 1, 13),
            cmap=boundary_cmap,
            alpha=0.42,
        )
        ax.contour(grid_x, grid_y, p, levels=[0.5], colors="#222222", linewidths=2.0)
        ax.axhline(
            0,
            color="#666666",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            label="true XOR boundary",
        )
        ax.axvline(
            0,
            color="#666666",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            label="_nolegend_",
        )
        ax.scatter(
            x_train[:, 0],
            x_train[:, 1],
            c=class_colors(y_train, class_purple, class_gold),
            edgecolor=train_color,
            linewidth=0.75,
            s=34,
            marker="o",
            label="training",
        )
        ax.scatter(
            x_validation[:, 0],
            x_validation[:, 1],
            c=class_colors(y_validation, class_purple, class_gold),
            edgecolor=validation_color,
            linewidth=1.05,
            marker="s",
            s=38,
            label="validation",
        )
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(-2.7, 2.7)
        ax.set_ylim(-2.7, 2.7)
        remove_outer_box(ax)
        ax.legend(frameon=False, loc="upper right", fontsize=8)

    def plot_decision_boundary(
        result,
        x_train,
        y_train,
        x_validation,
        y_validation,
        predict_proba,
        title,
        class_purple,
        class_gold,
        train_color,
        validation_color,
    ):
        fig, ax = plt.subplots(figsize=(6.7, 4.3))
        draw_boundary(
            ax,
            result,
            x_train,
            y_train,
            x_validation,
            y_validation,
            predict_proba,
            title,
            class_purple,
            class_gold,
            train_color,
            validation_color,
        )
        return fig

    def plot_boundary_comparison(
        raw_result,
        regularized_result,
        x_train,
        y_train,
        x_validation,
        y_validation,
        predict_proba,
        class_purple,
        class_gold,
        train_color,
        validation_color,
    ):
        fig, axes = plt.subplots(1, 2, figsize=(9.1, 4.0), sharex=True, sharey=True)
        draw_boundary(
            axes[0],
            raw_result,
            x_train,
            y_train,
            x_validation,
            y_validation,
            predict_proba,
            "No regularization",
            class_purple,
            class_gold,
            train_color,
            validation_color,
        )
        draw_boundary(
            axes[1],
            regularized_result,
            x_train,
            y_train,
            x_validation,
            y_validation,
            predict_proba,
            "Dropout + L2",
            class_purple,
            class_gold,
            train_color,
            validation_color,
        )
        axes[1].set_ylabel("")
        fig.suptitle("Same data and architecture", y=1.02, fontsize=13)
        fig.tight_layout()
        return fig

    def plot_loss_curves(result, title, train_color, validation_color, bayes_loss):
        epochs = np.arange(1, len(result["train_loss"]) + 1)
        fig, ax = plt.subplots(figsize=(6.7, 3.35))
        ax.plot(
            epochs,
            result["train_loss"],
            label="training loss",
            color=train_color,
            linewidth=2,
        )
        ax.plot(
            epochs,
            result["validation_loss"],
            label="validation loss",
            color=validation_color,
            linewidth=2,
        )
        ax.axvline(
            result["best_epoch"],
            color="#444444",
            linestyle=":",
            linewidth=1.5,
            label="best validation epoch",
        )
        ax.axhline(
            bayes_loss,
            color="#c9a227",
            linestyle="--",
            linewidth=1.7,
            label="Bayes loss",
        )
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("binary cross-entropy")
        remove_outer_box(ax)
        ax.legend(frameon=False, fontsize=9)
        return fig

    def plot_regularization_comparison(
        raw_result,
        regularized_result,
        train_color,
        validation_color,
        bayes_loss,
    ):
        fig, ax = plt.subplots(figsize=(6.7, 3.55))
        ax.plot(
            raw_result["validation_loss"],
            color=train_color,
            linewidth=2,
            label="validation, no regularization",
        )
        ax.plot(
            regularized_result["validation_loss"],
            color=validation_color,
            linewidth=2,
            linestyle="--",
            label="validation, dropout + L2",
        )
        ax.axhline(
            bayes_loss,
            color="#c9a227",
            linestyle="--",
            linewidth=1.7,
            label="Bayes loss",
        )
        ax.set_title("Validation loss under the same architecture")
        ax.set_xlabel("epoch")
        ax.set_ylabel("binary cross-entropy")
        remove_outer_box(ax)
        ax.legend(frameon=False, fontsize=9)
        return fig

    def model_summary_markdown(result, bayes):
        return "\n".join(
            [
                f"**Best = validation epoch {result['best_epoch']}.**",
                "",
                "|  | Bayes | Best | Train | Val |",
                "|---|---:|---:|---:|---:|",
                (
                    "| Accuracy | "
                    f"{percent(bayes['accuracy'])} | "
                    f"{percent(result['best_validation_accuracy'])} | "
                    f"{percent(result['train_accuracy'])} | "
                    f"{percent(result['validation_accuracy'])} |"
                ),
                (
                    "| Loss | "
                    f"{bayes['loss']:.3f} | "
                    f"{result['best_validation_loss']:.3f} | "
                    f"{result['final_train_loss']:.3f} | "
                    f"{result['final_validation_loss']:.3f} |"
                ),
                "",
                "Bayes = true XOR rule plus the known label-noise probability.",
            ]
        )

    return (
        bayes_metrics,
        model_summary_markdown,
        plot_boundary_comparison,
        plot_decision_boundary,
        plot_loss_curves,
        plot_regularization_comparison,
    )


@app.cell
def section_capacity(
    bayes_metrics,
    capacity_activation,
    capacity_epochs,
    capacity_layers,
    capacity_n,
    capacity_noise,
    capacity_units,
    class_gold,
    class_purple,
    compute_param_count,
    make_xor_data,
    model_summary_markdown,
    mo,
    parameter_breakdown,
    plot_decision_boundary,
    plot_loss_curves,
    predict_proba,
    train_color,
    train_mlp,
    train_validation_split,
    validation_color,
):
    _seed = 13
    _epochs = capacity_epochs.value
    _x, _y = make_xor_data(
        capacity_n.value,
        capacity_noise.value,
        _seed,
    )
    _x_train, _y_train, _x_validation, _y_validation = train_validation_split(
        _x,
        _y,
        0.4,
        _seed,
    )
    _param_count = compute_param_count(
        _x.shape[1],
        capacity_layers.value,
        capacity_units.value,
    )
    _result = train_mlp(
        _x_train,
        _y_train,
        _x_validation,
        _y_validation,
        hidden_layers=capacity_layers.value,
        hidden_units=capacity_units.value,
        activation=capacity_activation.value,
        epochs=_epochs,
        seed=_seed,
    )
    _bayes = bayes_metrics(_x_validation, _y_validation, capacity_noise.value)
    _boundary = plot_decision_boundary(
        _result,
        _x_train,
        _y_train,
        _x_validation,
        _y_validation,
        predict_proba,
        "Fitted boundary on XOR",
        class_purple,
        class_gold,
        train_color,
        validation_color,
    )
    _summary = mo.vstack(
        [
            mo.Html(
                f"""
                <div class="ml4p-param-box">
                  <div class="ml4p-param-number">{_param_count:,}</div>
                  <div>fitted parameters</div>
                </div>
                """
            ),
            mo.md(model_summary_markdown(_result, _bayes)),
        ]
    )
    _losses = plot_loss_curves(
        _result,
        "Training vs validation loss",
        train_color,
        validation_color,
        _bayes["loss"],
    )
    _param_breakdown = parameter_breakdown(
        _x.shape[1],
        capacity_layers.value,
        capacity_units.value,
    )
    capacity_bundle = {
        "x_train": _x_train,
        "y_train": _y_train,
        "x_validation": _x_validation,
        "y_validation": _y_validation,
        "result": _result,
        "param_count": _param_count,
        "hidden_layers": capacity_layers.value,
        "hidden_units": capacity_units.value,
        "activation": capacity_activation.value,
        "epochs": _epochs,
        "seed": _seed,
        "bayes_loss": _bayes["loss"],
    }
    mo.vstack(
        [
            mo.md(
                """
                ## Many Parameters, One Validation Set

                The data are a noisy XOR problem. First we apply the true XOR
                rule; then label noise flips a fraction of labels. The dashed
                lines at `x1 = 0` and `x2 = 0` show the correct boundary before
                those label flips.

                A flexible network can fit complicated boundaries, even with
                more fitted parameters than observations. The summary asks
                whether the fit travels to validation data.
                """
            ),
            mo.hstack([_boundary, _summary], gap=1.0, wrap=True),
            _losses,
            mo.md(_param_breakdown),
        ]
    )
    return capacity_bundle


@app.cell
def section_regularization(
    capacity_bundle,
    class_gold,
    class_purple,
    dropout_rate,
    l2_strength,
    mo,
    plot_boundary_comparison,
    plot_regularization_comparison,
    predict_proba,
    train_color,
    train_mlp,
    validation_color,
):
    _regularized = train_mlp(
        capacity_bundle["x_train"],
        capacity_bundle["y_train"],
        capacity_bundle["x_validation"],
        capacity_bundle["y_validation"],
        hidden_layers=capacity_bundle["hidden_layers"],
        hidden_units=capacity_bundle["hidden_units"],
        activation=capacity_bundle["activation"],
        epochs=capacity_bundle["epochs"],
        dropout_rate=dropout_rate.value,
        l2=l2_strength.value,
        seed=capacity_bundle["seed"],
    )
    _comparison = plot_regularization_comparison(
        capacity_bundle["result"],
        _regularized,
        train_color,
        validation_color,
        capacity_bundle["bayes_loss"],
    )
    _boundary_pair = plot_boundary_comparison(
        capacity_bundle["result"],
        _regularized,
        capacity_bundle["x_train"],
        capacity_bundle["y_train"],
        capacity_bundle["x_validation"],
        capacity_bundle["y_validation"],
        predict_proba,
        class_purple,
        class_gold,
        train_color,
        validation_color,
    )
    mo.vstack(
        [
            mo.md(
                """
                ---

                ## Regularization: Same Network, Different Training Rule

                This repeats the plots above with the **same XOR data split**
                and the **same architecture**: same layers, units, activation,
                and number of fitted parameters. Only the training rule changes.

                **Dropout** randomly removes hidden activations during training,
                so the network cannot rely too heavily on one hidden unit.
                **L2 regularization** (weight decay) penalizes large weights,
                so very sharp boundaries are harder to learn. Neither option
                reduces the number of fitted parameters; both change which
                parameter values are easy for the optimizer to choose.
                """
            ),
            _boundary_pair,
            _comparison,
        ]
    )
    return


@app.cell
def footer(mo):
    mo.Html(
        """
        <hr>
        <p><em>Javier Garcia-Bernardo -- SoDa & Utrecht University.</em></p>
        <p>Built with <a href="https://marimo.io/">Marimo</a>. This app uses
        only NumPy and Matplotlib so it can run as a standalone browser/WASM
        teaching app.</p>
        """
    )
    return


if __name__ == "__main__":
    app.run()
