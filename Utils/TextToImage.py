import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image

def TextToImage(text,fontname="Monospace",fontsize=24):    
    x = text[0].split('\n')

    size = (len(x),len(x[0]))

    plt.figure(figsize=size)
    plt.text(0,0, text, color='black',fontname=fontname, fontsize=fontsize)
    plt.axis('off')
    oid = io.BytesIO()
    plt.savefig(oid, format='png', bbox_inches='tight')
    oid.seek(0)
    img = plt.imread(oid)
    oid.close()
    plt.close()
    img = (img*255).astype(np.uint8)    
    img = Image.fromarray(img).convert('L')
    img = np.array(img)
    return  img


def unique_char_mapping(string):
    char_to_int = {}
    for char in string:
        if char not in char_to_int:
            char_to_int[char] = len(char_to_int)
    
    return char_to_int