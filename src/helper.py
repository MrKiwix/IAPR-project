import matplotlib.pyplot as plt

def display_sample(img, label, figsize=(12, 6), font_size=15):

    # Create side-by-side axes
    fig, (ax_img, ax_lbl) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]})

    # show image
    
    ax_img.imshow(img.numpy().transpose(1, 2, 0))
    ax_img.set_title("Sample Image")

    # show labels
    ax_lbl.axis("off")
    ax_lbl.set_title("Label Counts")
    text = "\n".join(f"{cls:20s} {cnt}" for cls, cnt in label.items())
    ax_lbl.text(0, 1, text, va="top", fontfamily="monospace", fontsize=font_size)

    plt.tight_layout()
    plt.show()
