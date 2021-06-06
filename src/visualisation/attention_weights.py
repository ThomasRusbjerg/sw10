import cv2, os
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor


def visualise_attention_weights(cfg, data):
    pred = DefaultPredictor(cfg)
    model = pred.model.detr
    file_name = os.path.basename(data[0]["file_name"])
    file_path = os.path.join("data/MUSCIMA++/v2.0/data/staves/images", file_name)
    img = cv2.imread(file_path)

    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    # propagate through the model
    pred(img)

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]

    # get the feature map shape
    h, w = conv_features['res5'].tensors.shape[-2:]
    create_decoder_cross_attention_figure(dec_attn_weights, h, w, img)
    # create_encoder_attention_figure(enc_attn_weights, h, w, img)


def create_decoder_cross_attention_figure(dec_attn_weights, h, w, img):
    fig = plt.figure()

    gs = fig.add_gridspec(2, 2)
    for i, query_idx in enumerate(range(0, 40, 20)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(dec_attn_weights[0, query_idx].view(h, w), cmap="Blues", interpolation="spline16")
        ax.set_title(f"query id: {query_idx}")
        ax.axis("off")

    ax4 = fig.add_subplot(gs[1, :])
    ax4.imshow(cv2.bitwise_not(img))
    ax4.set_title("input score")
    plt.axis("off")
    fig.tight_layout()

    save_path = "data/MUSCIMA++/v2.0/visualisations"
    image_name = "decoder-attention.jpg"
    fig.savefig(os.path.join(save_path, image_name))
    print(f"Saved figure of decoder cross-attention in {save_path}")
    # plt.show()


def create_encoder_attention_figure(enc_attn_weights, h, w, img):
    print(enc_attn_weights[0].shape)
    print((h, w) + (h, w))
    exit()
    # and reshape the self-attention to a more interpretable shape
    sattn = enc_attn_weights[0].reshape((h, w) + (h, w))
    print("Reshaped self-attention:", sattn.shape)
    # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
    fact = 32

    # let's select 4 reference points for visualization
    img_height = img.shape[0]
    img_width = img.shape[1]
    print(img_height)
    print(img_width)
    idxs = [(img_width//2, img_height//4), (img_width//2, img_height//2), (80, 90), (100, 110)]
    # idxs = [(100, 150), (100, 160), (150, 150), (100, 170)]

    # here we create the canvas
    fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
    # and we add one plot per reference point
    gs = fig.add_gridspec(2, 4)
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[0, -1]),
        fig.add_subplot(gs[1, -1]),
    ]

    # for each one of the reference points, let's plot the self-attention
    # for that point
    for idx_o, ax in zip(idxs, axs):
        print(idx_o)
        idx = (idx_o[0] // fact, idx_o[1] // fact)
        print(idx)
        ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'self-attention{idx_o}')

    # and now let's add the central image, with the reference points as red circles
    fcenter_ax = fig.add_subplot(gs[:, 1:-1])
    fcenter_ax.imshow(img)
    for (y, x) in idxs:
        scale = img.shape[-2] / img.shape[-1]
        x = ((x // fact) + 0.5) * fact
        y = ((y // fact) + 0.5) * fact
        fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
        fcenter_ax.axis('off')

    save_path = "data/MUSCIMA++/v2.0/visualisations"
    image_name = "encoder-attention.jpg"
    fig.savefig(os.path.join(save_path, image_name))
    print(f"Saved figure of encoder self-attention in {save_path}")
    # plt.show()
