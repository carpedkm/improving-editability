import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import clip
import json
import argparse
from transformers import AutoProcessor, LlavaForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from multiedit_dataset import multiedit_DATASET


def evaluate_clip_layerwise_score(path_to_generated: str, val_loader: torch.utils.data.DataLoader, 
                                args: argparse.Namespace, dataset: multiedit_DATASET) -> tuple[float, float, float, float]:
    """
    Evaluate layer-wise CLIP scores for generated images.

    Args:
        path_to_generated (str): Path to directory containing generated images.
        val_loader (DataLoader): DataLoader for validation dataset.
        args (Namespace): Command-line arguments containing GPU index and result directory.
        dataset (MultieditDataset): Dataset object for index mapping.

    Returns:
        tuple[float, float, float, float]: Mean CLIP scores for class and prompt, and their standard deviations.
    """
    model, preprocess = clip.load('ViT-B/32', device=f'cuda:{args.GPU_IDX}')
    model = model.to(f'cuda:{args.GPU_IDX}')
    
    clip_scores_class_all = []
    clip_scores_prompt_all = []
    os.makedirs(os.path.join(args.result_dir, 'cropped_images'), exist_ok=True)

    for idx, data in tqdm(enumerate(val_loader), total=len(val_loader), desc="Evaluating layer-wise CLIP scores"):
        layers = int(data['total_layers'])
        orig_idx = dataset.idx_mapper[str(idx)]
        image_path = os.path.join(path_to_generated, f'IDX{orig_idx}')
        
        images = [Image.open(os.path.join(image_path, f'IDX{orig_idx}_layer{i}.png')).convert('RGB') 
                 for i in range(layers)]
        
        bboxes_pixel = [[int(coord) for coord in bbox] for bbox in data['bboxes_pixel']]
        cropped_images = [np.array(image) * np.stack([bbox.numpy()] * 3, axis=-1) 
                         for image, bbox in zip(images, data['bboxes'].squeeze())]
        cropped_images = [Image.fromarray(image[bbox[0]:bbox[2], bbox[1]:bbox[3], ...].astype(np.uint8)).resize((224, 224)) 
                         for bbox, image in zip(bboxes_pixel, cropped_images)]

        concatenated_cropped = np.concatenate([np.array(img)[:, :, ::-1] for img in cropped_images], axis=1)
        Image.fromarray(concatenated_cropped).save(
            os.path.join(args.result_dir, 'cropped_images', f'IDX{orig_idx}_layerwise_clip.png'))

        cropped_images = torch.stack([preprocess(img) for img in cropped_images]).to(f'cuda:{args.GPU_IDX}')
        
        with torch.no_grad():
            encoded_images = model.encode_image(cropped_images)
            encoded_images = encoded_images / encoded_images.norm(dim=-1, keepdim=True)

        class_texts = [f"An image of {data['classes'][i]} in {data['background'][0]}" for i in range(layers)]
        tokenized_texts_c = clip.tokenize(class_texts).to(f'cuda:{args.GPU_IDX}')
        with torch.no_grad():
            encoded_texts_c = model.encode_text(tokenized_texts_c)
            encoded_texts_c = encoded_texts_c / encoded_texts_c.norm(dim=-1, keepdim=True)

        prompt_texts = [data['local_prompts'][i][0][:77] for i in range(layers)]
        tokenized_texts_p = clip.tokenize(prompt_texts).to(f'cuda:{args.GPU_IDX}')
        with torch.no_grad():
            encoded_texts_p = model.encode_text(tokenized_texts_p)
            encoded_texts_p = encoded_texts_p / encoded_texts_p.norm(dim=-1, keepdim=True)

        clip_scores_class = 100.0 * encoded_images @ encoded_texts_c.T
        clip_scores_prompt = 100.0 * encoded_images @ encoded_texts_p.T
        
        clip_score_class_scores = np.array([clip_scores_class[i, i].cpu() for i in range(clip_scores_class.shape[0])])
        clip_score_prompt_scores = np.array([clip_scores_prompt[i, i].cpu() for i in range(clip_scores_prompt.shape[0])])
        
        clip_scores_class_all.append(clip_score_class_scores.mean().item())
        clip_scores_prompt_all.append(clip_score_prompt_scores.mean().item())

    with open(os.path.join(args.result_dir, 'clip_scores.txt'), 'w') as f:
        for i, (c, p) in enumerate(zip(clip_scores_class_all, clip_scores_prompt_all)):
            f.write(f"IDX{i}: class={c}, prompt={p}\n")

    return (np.mean(clip_scores_class_all), np.mean(clip_scores_prompt_all), 
            np.std(clip_scores_class_all), np.std(clip_scores_prompt_all))


def sentence_meteor(reference: str, hypothesis: str) -> float:
    """Calculate METEOR score for a single reference and hypothesis."""
    return meteor_score([reference.split()], hypothesis.split())

def sentence_rouge(reference: str, hypothesis: str) -> float:
    """Calculate ROUGE-L score for a single reference and hypothesis."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(reference, hypothesis)['rougeL'].fmeasure

def evaluate_llava_layerwise_score(path_to_generated: str, val_loader: torch.utils.data.DataLoader, 
                                 args: argparse.Namespace, dataset: multiedit_DATASET) -> tuple[dict, float, float, float, float]:
    """
    Evaluate layer-wise LLAVA-generated captions using BLEU, CIDEr, METEOR, and ROUGE scores.

    Args:
        path_to_generated (str): Path to directory containing generated images.
        val_loader (DataLoader): DataLoader for validation dataset.
        args (Namespace): Command-line arguments containing GPU index and result directory.
        dataset (multiedit_DATASET): Dataset object for index mapping.

    Returns:
        tuple[dict, float, float, float, float]: BLEU scores (1-4), METEOR, ROUGE scores, and METEOR std-dev.
    """
    bleu_weights = {1: (1, 0, 0, 0), 2: (0.5, 0.5, 0, 0), 3: (0.33, 0.33, 0.33, 0), 4: (0.25, 0.25, 0.25, 0.25)}
    
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-13b-hf").half().to(f'cuda:{args.GPU_IDX}')
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")

    prompt = ("USER: <image>\nCan you briefly describe this image in a few words? "
              "If it helps, make it precise. Include only necessary points. "
              "It is necessary to include the shape of the object or the object itself with the amount of it. "
              "If there is only one object, include only the object itself. "
              "Keep it factual with no subjective speculations. Start with the words 'This image showcases'. ASSISTANT:")
    
    all_bleu_scores = {1: [], 2: [], 3: [], 4: []}
    all_meteor_scores = []
    all_rouge_scores = []
    os.makedirs(os.path.join(args.result_dir, 'cropped_images'), exist_ok=True)

    for idx, data in tqdm(enumerate(val_loader), total=len(val_loader), desc="Evaluating layer-wise LLAVA scores"):
        layers = int(data['total_layers'])
        orig_idx = dataset.idx_mapper[str(idx)]
        images = [Image.open(os.path.join(path_to_generated, f'IDX{orig_idx}', f'IDX{orig_idx}_layer{i}.png')).convert('RGB') 
                 for i in range(layers)]

        bboxes_pixel = [[int(coord) for coord in bbox] for bbox in data['bboxes_pixel']]
        cropped_images = [np.array(image) * np.stack([bbox.numpy()] * 3, axis=-1) 
                         for image, bbox in zip(images, data['bboxes'].squeeze())]
        cropped_images = [Image.fromarray(image[bbox[0]:bbox[2], bbox[1]:bbox[3], ...].astype(np.uint8)).resize((224, 224)) 
                         for bbox, image in zip(bboxes_pixel, cropped_images)]

        concatenated_cropped = np.concatenate([np.array(img) for img in cropped_images], axis=1)
        Image.fromarray(concatenated_cropped).save(
            os.path.join(args.result_dir, 'cropped_images', f'IDX{orig_idx}_layerwise_llava.png'))

        processed_texts = []
        for cropped_image in cropped_images:
            inputs = processor(text=prompt, images=cropped_image, return_tensors="pt").to(f'cuda:{args.GPU_IDX}')
            with torch.cuda.amp.autocast(dtype=torch.float16):
                generate_ids = model.generate(**inputs, max_new_tokens=512)
            generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            generated_caption = generated_text.split('ASSISTANT:')[1].strip()
            processed_texts.append(generated_caption)

        with open(os.path.join(args.result_dir, 'generated_captions.txt'), 'a') as f:
            f.write(f"IDX{orig_idx}\n{'\n'.join(processed_texts)}\n------\n")

        for i in range(layers):
            reference = data['local_prompts'][i][0]
            hypothesis = processed_texts[i]
            
            for n in range(1, 5):
                all_bleu_scores[n].append(sentence_bleu([reference], hypothesis, weights=bleu_weights[n]))

            all_meteor_scores.append(sentence_meteor(reference, hypothesis))
            all_rouge_scores.append(sentence_rouge(reference, hypothesis))

    with open(os.path.join(args.result_dir, 'bleu_scores.txt'), 'w') as f:
        for i in range(len(all_bleu_scores[1])):
            f.write(f"IDX{i}: {all_bleu_scores[1][i]}, {all_bleu_scores[2][i]}, {all_bleu_scores[3][i]}, {all_bleu_scores[4][i]}\n")
    
    
    with open(os.path.join(args.result_dir, 'meteor_scores.txt'), 'w') as f:
        for i, meteor in enumerate(all_meteor_scores):
            f.write(f"IDX{i}: {meteor}\n")
    
    with open(os.path.join(args.result_dir, 'rouge_scores.txt'), 'w') as f:
        for i, rouge in enumerate(all_rouge_scores):
            f.write(f"IDX{i}: {rouge}\n")

    overall_bleu_score = {n: np.mean(all_bleu_scores[n]) for n in range(1, 5)}
    overall_meteor_score = np.mean(all_meteor_scores)
    overall_rouge_score = np.mean(all_rouge_scores)
    meteor_std = np.std(all_meteor_scores)

    return overall_bleu_score, overall_meteor_score, overall_rouge_score, meteor_std

def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-edit dataset using CLIP and LLAVA metrics")
    parser.add_argument('--result_dir', type=str, default='path-for-results', 
                       help='Directory containing generated images')
    parser.add_argument('--dataset_json', type=str, 
                       default='path-to-json', 
                       help='Path to dataset JSON file')
    parser.add_argument('--GPU_IDX', type=int, default=1, help='GPU index')
    args = parser.parse_args()

    with open(args.dataset_json, 'r') as f:
        loaded_json = json.load(f)
    loaded_json = {k: loaded_json[k] for k in list(loaded_json.keys())[:100]}
    
    dataset = multiedit_DATASET(loaded_json, in_pipeline=False)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    clip_score_c, clip_score_p, std_c, std_p = evaluate_clip_layerwise_score(
        os.path.join(args.result_dir, 'gen'), val_loader, args, dataset)
    print(f"Clip score (class): {clip_score_c:.4f}")
    print(f"Clip score std-dev (class): {std_c:.4f}")
    print(f"Clip score (prompt): {clip_score_p:.4f}")
    print(f"Clip score std-dev (prompt): {std_p:.4f}")
    bleu, meteor, rouge, meteor_std = evaluate_llava_layerwise_score(
        os.path.join(args.result_dir, 'gen'), val_loader, args, dataset)
    print(f"BLEU 1/2/3/4 score: {bleu[1]:.4f} / {bleu[2]:.4f} / {bleu[3]:.4f} / {bleu[4]:.4f}")
    print(f"METEOR score: {meteor:.4f}")
    print(f"METEOR std-dev: {meteor_std:.4f}")
    print(f"ROUGE score: {rouge:.4f}")

if __name__ == '__main__':
    main()