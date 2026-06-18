from typing import Optional
import math
import matplotlib.pyplot as plt
import molplotly
import numpy as np
import pandas as pd
from plotly import express as px

from networking import check_port_in_use, get_free_port, get_url_for_port


def plot_data(
    data: np.ndarray,
    matched_compounds: pd.DataFrame,
    color_col: str = "Superclass",
    marker_size: float = 2.5,
    debug: bool = False,
    plot_title: str = "Projection",
    host: str = "0.0.0.0",
    url: Optional[str] = None,
    db_name: Optional[str] = None,
):
    df = pd.DataFrame(data, columns=["X", "Y"])
    df = pd.concat([df, matched_compounds.reset_index(drop=True)], axis=1)

    fig = px.scatter(
        df,
        x="X",
        y="Y",
        height=600,
        width=800,
        color=color_col,
        title=f"{plot_title}: {db_name}" if db_name else plot_title,
    )
    fig.update_traces(marker=dict(size=marker_size))

    app = molplotly.add_molecules(
        fig=fig,
        df=df,
        smiles_col="SMILES",
        title_col="inchikey",
        color_col=color_col,
        caption_cols=df.columns.difference(
            ["X", "Y", "SMILES", "inchikey", "Kingdom"]
        ).tolist(),
    )
    free_port = get_free_port(host=host)
    if not check_port_in_use(free_port):
        app.run(host=host, port=free_port, debug=debug)

    print(
        f"Plot running on port {free_port}",
        get_url_for_port(free_port, host=url if url else host),
    )


def export_plot_grid(
    embedding: np.ndarray,
    matched_compounds: pd.DataFrame,
    color_cols: list[str],
    filename: str,
    plot_title: str = "Projection",
    n_cols: Optional[int] = None,
    panel_size: float = 6.0,
    point_size: float = 3.0,
):
    embedding = np.asarray(embedding)

    n_plots = len(color_cols)
    if n_plots == 0:
        raise ValueError("color_cols is empty; nothing to plot.")

    if n_cols is None:
        n_cols = min(n_plots, math.ceil(math.sqrt(n_plots)))
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_size * n_cols, panel_size * n_rows),
        layout="constrained",
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for ax, color_col in zip(flat_axes, color_cols):
        labels = matched_compounds[color_col]
        codes = (
            labels
            if pd.api.types.is_numeric_dtype(labels)
            else pd.Categorical(labels).codes
        )
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=codes,
            cmap="Spectral",
            s=point_size,
            linewidths=0,
        )
        ax.set_aspect("equal")  # fills the column width while keeping geometry honest
        ax.set(title=color_col, xlabel="x", ylabel="y", xticks=[], yticks=[])

    for ax in flat_axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle(plot_title, fontsize=16, fontweight="bold")

    fig.savefig(filename, dpi=300)
    plt.show()

    print(f"Saved {n_plots} panel(s) in a {n_rows}x{n_cols} grid to: {filename}")
