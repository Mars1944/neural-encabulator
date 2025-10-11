import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def plot_training_results(loss_history, accuracy_history, hyperparams, output_dir="plots"):
    """
    Plot training loss and accuracy over epochs, including:
      - Best polynomial trendline for accuracy (order ≤ 10)
      - R² value and polynomial equation displayed on the plot
      - Hyperparameters printed in title and filename
    """

    os.makedirs(output_dir, exist_ok=True)
    epochs = np.arange(1, len(loss_history) + 1)

    # ===============================================================
    # === Find best polynomial fit for accuracy ===
    # ===============================================================
    y = np.array(accuracy_history)
    best_r2 = -np.inf
    best_order = 1
    best_fit = None
    best_poly = None

    for order in range(1, 11):
        coeffs = np.polyfit(epochs, y, order)
        poly = np.poly1d(coeffs)
        y_fit = poly(epochs)
        r2 = r2_score(y, y_fit)
        if r2 > best_r2:
            best_r2 = r2
            best_order = order
            best_fit = y_fit
            best_poly = poly

    print(f"Best polynomial fit: order={best_order}, R²={best_r2:.4f}")
    print(f"Polynomial coefficients (highest → lowest order): {best_poly.coefficients}")

    # ===============================================================
    # === Create equation string for annotation ===
    # ===============================================================
    # Example: y = 0.01x² - 0.3x + 85.7
    terms = []
    for i, c in enumerate(best_poly.coefficients):
        power = best_order - i
        if power > 1:
            terms.append(f"{c:+.3e}x^{power}")
        elif power == 1:
            terms.append(f"{c:+.3e}x")
        else:
            terms.append(f"{c:+.3e}")
    poly_eq = " ".join(terms)
    eq_str = f"y = {poly_eq}\nR² = {best_r2:.4f}"

    # ===============================================================
    # === Create plot ===
    # ===============================================================
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, loss_history, color='tab:blue', label='Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)

    # Accuracy (right axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:orange')
    ax2.plot(epochs, accuracy_history, color='tab:orange', label='Accuracy', linewidth=2)
    ax2.plot(epochs, best_fit, '--', color='tab:red', linewidth=2.5,
             label=f'Best Fit (Order {best_order})')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # ===============================================================
    # === Title, legend, and annotation ===
    # ===============================================================
    hyper_str = ', '.join([f'{k}={v}' for k, v in hyperparams.items()])
    plt.title(f"Training Results\n({hyper_str})\nBest Polynomial Fit: Order {best_order}")

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # Add polynomial equation text
    #plt.text(0.02, 0.05, eq_str,
    #         transform=ax1.transAxes,
    #         fontsize=9,
    #         color='darkred',
    #         verticalalignment='bottom',
    #         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=1))

    fig.tight_layout()

    # ===============================================================
    # === Save figure ===
    # ===============================================================
    filename = f"training_plot_{'_'.join([f'{k}{v}' for k, v in hyperparams.items()])}.png"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"📊 Plot saved to: {plot_path}")
    return plot_path
