import numpy as np
from PIL import Image, ImageTk
from tkinter import Canvas, Tk

def compress_channel(channel, k, gain=5.0):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    S_noise = np.zeros_like(S)
    S_noise[k:] = S[k:]  #for noise isolation
    
    reconstructed = np.dot(U * S_noise, Vt) * gain + 128
    return reconstructed

def main():
    try:
        img_path = "image.jpg"
        image = Image.open(img_path).convert('RGB')
        pixel_array = np.asarray(image)
    except FileNotFoundError:
        print(f"Error: Could not find '{img_path}'")
        return

    root = Tk()
    root.title("SVD Image Compression")
    h, w, _ = pixel_array.shape
    canvas = Canvas(root, width=w, height=h, bg="white")
    canvas.pack()

    k = 50
    
    compressed_channels = []
    for i in range(3):
        channel = pixel_array[:, :, i]
        compressed = compress_channel(channel, k)
        compressed_channels.append(compressed)

    final_array = np.stack(compressed_channels, axis=2)
    final_array = np.clip(final_array, 0, 255).astype(np.uint8)

    tk_image = ImageTk.PhotoImage(image=Image.fromarray(final_array))
    canvas.create_image(0, 0, anchor="nw", image=tk_image)

    canvas.image = tk_image 

    print(f"Displaying compressed image with rank k={k}")
    root.mainloop()

if __name__ == "__main__":
    main()
