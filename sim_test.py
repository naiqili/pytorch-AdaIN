import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image

import sim_net as net
from function import adaptive_instance_normalization
from function import coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, network, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    loss_c, loss_s = network(content)
    style_mean, style_std = network.simulate()
    t = network.adain(content_f, style_mean[-1], style_std[-1])
    feat = alpha * t + (1 - alpha) * content_f
    return decoder(feat)


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)

if args.content:
    content_paths = [args.content]
else:
    content_paths = [os.path.join(args.content_dir, f) for f in
                     os.listdir(args.content_dir)]


if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)
network = net.SimNet(vgg, decoder, device)
network.to(device)

content_tf = test_transform(args.content_size, args.crop)

for content_path in content_paths:
    content = content_tf(Image.open(content_path))
    content = content.to(device).unsqueeze(0)
    with torch.no_grad():
        output = style_transfer(vgg, decoder, content, network,
                                args.alpha)
    output = output.cpu()

    output_name = '{:s}/{:s}_stylized{:s}'.format(
        args.output, splitext(basename(content_path))[0],
        args.save_ext
    )
    save_image(output, output_name)
