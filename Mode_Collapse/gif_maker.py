import glob
from PIL import Image
def make_gif(frame_folder):
    frames = []
    for i in range(50):
        if i < 9:
            frames.append(Image.open(f"{frame_folder}/0"+ str(10*(1+i))+ ".png"))
        else:
            frames.append(Image.open(f"{frame_folder}/" + str(10*(1+i))+ ".png"))
    # frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    # print(frames)
    frame_one = frames[0]
    frame_one.save("repJSD.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
    
if __name__ == "__main__":
    make_gif("./plots/repJSD")