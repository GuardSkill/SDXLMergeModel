from infer import Inference
import os 
import gc
import torch

promts=[
     '1girl, white woman,  hyper realistic, good figure, beautiful, wear vest, upper-body, look at viewer, complex background',
     'photorealism, intricate details, extremely detailed, masterpiece, masterpiece, Outstanding intricacies, best quality, hires textures, high detail, incredibly detailed, Cinematic Lighting, at night, Cinematic photography, movie mood, cinematic light, compelling composition, storytelling elements, conveys emotion, mood, and narrative depth, creating visually striking images that feel like still frames from a film',
     '1girl, cartoon, good figure, beautiful, wear vest, upper-body, look at viewer, ride bycycle, complex background',
     '1girl, anime, good figure, beautiful, wear vest, upper-body, look at viewer, walk in road, complex background',
     'HuracánCar,sports car,white car,masterpiece, front isometric view,neon lights tires, cyberpunk background, smooth outer texture ,bokeh,cinematic shot,big tires',
     '(best quality, 8K, high resolution, masterpiece), ultra detailed, (3D CGI), a beautiful pair of sunglasses, trendy, fashionable, pink, silver and black stylized Cat-eye shape, on a black background, countryside advertising, winning photo'
     'a brown puppy lies on the ground and looks at the audience, happy, very cute, close-up, professional photography',
     'score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, source_furry, a cute bunny',
     'Artist at work drawing a Landscape with a pencil, Half way through, hand and pencil in colour, landscape black and white, pencil drawing on white paper, hand holding pencil drawing the landscape, Sketch, paper texture, Drawing,'
     'hyper realistic medium full shot photo of a (magical scenery:1.1), (oppulent , esthetic , masterful:1.4), poster art, bold lines, hyper detailed, expressive, award winning, dark limited color palette, high contrast, depth of field, (intricate details, masterpiece, best quality:1.4), fluorescent lighting, looking at viewer, dynamic pose, wide angle panoramic view,',
     '(best quality, 8K, high resolution, masterpiece), ultra detailed, a beautiful countryside ,winning photo',
     '(((photographic, photo, photogenic))), extremely high quality high detail RAW color photo, exquisite details of a shimmering and reflective pink gem rune stone are delicately highlighted against the contrasting darkness, a softly glowing celestial being, radiating a gentle light from its ethereal form, its luminous plasma streamline with an otherworldly power, adding to its captivating presence, mesmerizing allure and ethereal nature, divine proportions, luxury ambiance, Canon EOS R10 effect, focus on art by @DrDB, whimsical lighting, cool tones, depth of field effect, intimate atmosphere, elegant artifact gem , candid moment, background lights bokeh, ardent depth, meticulous details',
     '(Wuxia:1.5),(Surrealism:1.5),epic composition,stunning intricate details,glowing,dragons and horses,verdant mountains and rivers,heroes roam,masterpiece,best masterpiece,'
]

disable_torch_compile=True
sdxl=False
model_path=r'E:\Models\SD1.5\all\DreamShaper_8_pruned.safetensors'
output_path='./output/'

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

if not os.path.exists(output_path):
    os.makedirs(output_path)
    
index=0
for prompt in promts:
    index+=1
    infer = Inference(model_path, sdxl=sdxl, disable_torch_compile=disable_torch_compile)
    reso=[512,512]
    images = infer.validate(prompt, sdxl=sdxl,resolution=reso)
    free_memory()

    if output_path:
        basename = os.path.splitext(os.path.basename(model_path))[0]
        dirpath = output_path
        
        if isinstance(images, list):
            for idx, img in enumerate(images):
                img.save(os.path.join(dirpath, f'{basename}_{index}_{idx}.png'))
        else:
            images.save(os.path.join(dirpath, f'{basename}_{index}.png'))
