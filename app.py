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
        </style>
        """
    )
    return class_gold, class_purple, css, train_color, validation_color


@app.cell
def capacity_controls(mo):
    capacity_n = mo.ui.slider(40, 200, step=20, value=100, label="Sample size")
    capacity_noise = mo.ui.slider(0.0, 0.8, step=0.05, value=0.2, label="Noise")
    capacity_layers = mo.ui.slider(1, 6, step=1, value=3, label="Hidden layers")
    capacity_units = mo.ui.slider(2, 128, step=2, value=32, label="Units per layer")
    capacity_activation = mo.ui.dropdown(
        options=["ReLU", "tanh", "linear"],
        value="ReLU",
        label="Activation",
    )
    dropout_rate = mo.ui.slider(0.0, 0.6, step=0.05, value=0.25, label="Dropout rate")
    l1_strength = mo.ui.slider(0.0, 0.02, step=0.001, value=0.0, label="L1 penalty")
    l2_strength = mo.ui.slider(0.0, 0.05, step=0.001, value=0.005, label="L2 penalty")
    return (
        capacity_activation,
        capacity_layers,
        capacity_n,
        capacity_noise,
        capacity_units,
        dropout_rate,
        l1_strength,
        l2_strength,
    )


@app.cell
def sidebar(
    capacity_activation,
    capacity_layers,
    capacity_n,
    capacity_noise,
    capacity_units,
    dropout_rate,
    l1_strength,
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
            capacity_activation,
            mo.md("**Regularization**"),
            dropout_rate,
            l1_strength,
            l2_strength,
            mo.md(
                """
                <span class="ml4p-small">The second fit keeps the same data,
                architecture, initial weights, and 100 epochs; only
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
            mo.Html(
                """
                <div class="ml4p-callout">
                This app keeps one dataset fixed: <strong>XOR</strong>. The
                point is to compare what the network can fit with what it can
                predict on held-out data.
                </div>
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
        score = x[:, 0] * x[:, 1]
        y = (score + rng.normal(0, noise, size=n) > 0).astype(int)
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
        l1=0.0,
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
        best_loss = np.inf
        best_epoch = 1

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
                    grad_w[i] += l1 * np.sign(model["weights"][i])
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

            if validation_loss[-1] < best_loss - 1e-4:
                best_loss = validation_loss[-1]
                best_epoch = epoch + 1

        final_train = forward_pass(model, x_train, training=False)[0]
        final_validation = forward_pass(model, x_validation, training=False)[0]
        return {
            "model": model,
            "train_loss": np.array(train_loss),
            "validation_loss": np.array(validation_loss),
            "final_train_loss": binary_cross_entropy(y_train, final_train),
            "final_validation_loss": binary_cross_entropy(y_validation, final_validation),
            "best_epoch": best_epoch,
            "train_accuracy": accuracy(model, x_train, y_train),
            "validation_accuracy": accuracy(model, x_validation, y_validation),
        }

    return compute_param_count, parameter_breakdown, predict_proba, train_mlp


@app.cell
def plotting_helpers(np, plt):
    from matplotlib.colors import LinearSegmentedColormap

    def class_colors(y, class_purple, class_gold):
        colors = np.array([class_purple, class_gold])
        return colors[y.astype(int)]

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
        ax.legend(frameon=False, loc="upper right", fontsize=9)

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
        metrics_title=None,
        metrics_rows=None,
    ):
        if metrics_rows is None:
            fig, ax = plt.subplots(figsize=(6.7, 4.3))
            table_ax = None
        else:
            fig = plt.figure(figsize=(8.4, 4.15))
            spec = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.35], wspace=0.25)
            ax = fig.add_subplot(spec[0, 0])
            table_ax = fig.add_subplot(spec[0, 1])

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
        if table_ax is not None:
            table_ax.axis("off")
            table_ax.set_title(metrics_title, loc="left", pad=8, fontsize=12, fontweight="bold")
            table = table_ax.table(
                cellText=metrics_rows,
                cellLoc="left",
                colWidths=[0.63, 0.37],
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.35)
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor("#dddddd")
                cell.set_linewidth(0.6)
                cell.set_facecolor("white")
                if col == 0:
                    cell.set_text_props(color="#666666", weight="normal")
                else:
                    cell.set_text_props(color="#333333", weight="bold")
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
            "Dropout + L1/L2",
            class_purple,
            class_gold,
            train_color,
            validation_color,
        )
        axes[1].set_ylabel("")
        fig.suptitle("Same data and architecture", y=1.02, fontsize=13)
        fig.tight_layout()
        return fig

    def plot_loss_curves(result, title, train_color, validation_color):
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
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("binary cross-entropy")
        ax.legend(frameon=False, fontsize=9)
        return fig

    def plot_regularization_comparison(raw_result, regularized_result, train_color, validation_color):
        fig, ax = plt.subplots(figsize=(6.7, 3.55))
        ax.plot(
            raw_result["train_loss"],
            color=train_color,
            linewidth=2,
            alpha=0.55,
            label="training, no regularization",
        )
        ax.plot(
            regularized_result["train_loss"],
            color=train_color,
            linestyle=":",
            linewidth=2,
            label="training, regularized",
        )
        ax.plot(
            raw_result["validation_loss"],
            color=validation_color,
            linewidth=2,
            alpha=0.55,
            label="validation, no regularization",
        )
        ax.plot(
            regularized_result["validation_loss"],
            color=validation_color,
            linewidth=2,
            linestyle="--",
            label="validation, regularized",
        )
        ax.set_title("Same network and data; regularization changes training")
        ax.set_xlabel("epoch")
        ax.set_ylabel("binary cross-entropy")
        ax.legend(frameon=False, fontsize=9)
        return fig

    def make_metric_rows(result, param_count, n_train, n_validation):
        loss_gap = result["final_validation_loss"] - result["final_train_loss"]
        accuracy_gap = result["train_accuracy"] - result["validation_accuracy"]
        return [
            ["Parameters", f"{param_count:,}"],
            ["Train / val n", f"{n_train} / {n_validation}"],
            ["Train acc.", f"{result['train_accuracy']:.2f}"],
            ["Val acc.", f"{result['validation_accuracy']:.2f}"],
            ["Train loss", f"{result['final_train_loss']:.3f}"],
            ["Val loss", f"{result['final_validation_loss']:.3f}"],
            ["Loss gap", f"{loss_gap:.3f}"],
            ["Acc. gap", f"{accuracy_gap:.2f}"],
        ]

    return (
        make_metric_rows,
        plot_boundary_comparison,
        plot_decision_boundary,
        plot_loss_curves,
        plot_regularization_comparison,
    )


@app.cell
def section_capacity(
    capacity_activation,
    capacity_layers,
    capacity_n,
    capacity_noise,
    capacity_units,
    class_gold,
    class_purple,
    compute_param_count,
    make_metric_rows,
    make_xor_data,
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
    _epochs = 100
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
    _metrics = make_metric_rows(
        _result,
        _param_count,
        len(_y_train),
        len(_y_validation),
    )
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
        "Model summary",
        _metrics,
    )
    _losses = plot_loss_curves(
        _result,
        "Training vs validation loss",
        train_color,
        validation_color,
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
    }
    mo.vstack(
        [
            mo.md(
                """
                ## Many Parameters, One Validation Set

                A flexible network can fit complicated boundaries, even with
                more fitted parameters than observations. The table prints the
                fitted parameter count; the loss curve asks whether the fit
                travels to validation data.
                """
            ),
            _boundary,
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
    l1_strength,
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
        l1=l1_strength.value,
        l2=l2_strength.value,
        seed=capacity_bundle["seed"],
    )
    _comparison = plot_regularization_comparison(
        capacity_bundle["result"],
        _regularized,
        train_color,
        validation_color,
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

                ## Add Regularization To The Same Network

                This repeats the plots above with the **same XOR data split**
                and the **same architecture**: same layers, units, activation,
                and number of fitted parameters. Only the training rule changes.

                Dropout thins the hidden layers during training. L1 and L2 add
                penalties to the weights. None of these reduces the number of
                fitted parameters; they change which parameter values are easy
                for the optimizer to choose.
                """
            ),
            mo.md(
                f"""
                Regularized run: **{capacity_bundle["param_count"]:,}** fitted
                parameters, **dropout = {dropout_rate.value:.2f}**,
                **L1 = {l1_strength.value:.3f}**, **L2 = {l2_strength.value:.3f}**,
                same initial weights, same train/validation split, same 100
                epochs.
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
