import sys
sys.path.insert(0, ".")
import part1, part2, part3, part4
from kit import make_doc, S
doc = make_doc("Study_Notes_UNet_VAE_GAN_StableDiffusion.pdf")
doc.build(S)
print("built")
