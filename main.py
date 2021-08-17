import json
import logging
import os
import argparse
from collections import defaultdict
import random
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import WEIGHTS_NAME, AutoTokenizer

logger = logging.getLogger(__name__)

# from luke.utils.entity_vocab import MASK_TOKEN
MASK_TOKEN = "[MASK]"

from trainer import Trainer
from models.modeling_roberta import RobertaForEntitySpanQA
from models.modeling_electra import ElectraForEntitySpanQA
from models.modeling_bert import BertForEntitySpanQA
from record_eval import evaluate as evaluate_on_record
from utils import (
    HIGHLIGHT_TOKEN,
    PLACEHOLDER_TOKEN,
    ENTITY_MARKER_TOKEN,
    RecordProcessor,
    convert_examples_to_features,
    Writer
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def initialize_model(args):
    if args.lm_type == "roberta":
        model = RobertaForEntitySpanQA.from_pretrained("roberta-base")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        args.tokenizer = tokenizer
    
        word_emb = model.roberta.embeddings.word_embeddings.weight
        highlight_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)
        placeholder_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["#"])[0]].unsqueeze(0)
        marker_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["*"])[0]].unsqueeze(0)

        model.roberta.embeddings.word_embeddings.weight.data = torch.cat(
            [word_emb, highlight_emb, placeholder_emb, marker_emb]
        )
    elif args.lm_type == "electra":
        model = ElectraForEntitySpanQA.from_pretrained("google/electra-small-discriminator")
        tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
        args.tokenizer = tokenizer
    
        word_emb = model.electra.embeddings.word_embeddings.weight
        highlight_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)
        placeholder_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["#"])[0]].unsqueeze(0)
        marker_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["*"])[0]].unsqueeze(0)

        model.electra.embeddings.word_embeddings.weight.data = torch.cat(
            [word_emb, highlight_emb, placeholder_emb, marker_emb]
        )
    elif args.lm_type == "electra-base":
        model = ElectraForEntitySpanQA.from_pretrained("google/electra-base-discriminator")
        tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
        args.tokenizer = tokenizer
    
        word_emb = model.electra.embeddings.word_embeddings.weight
        highlight_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)
        placeholder_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["#"])[0]].unsqueeze(0)
        marker_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["*"])[0]].unsqueeze(0)

        model.electra.embeddings.word_embeddings.weight.data = torch.cat(
            [word_emb, highlight_emb, placeholder_emb, marker_emb]
        )
    elif args.lm_type == "bert-erica":
        model = BertForEntitySpanQA.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        args.tokenizer = tokenizer

        # Load ERICA checkpoint
        print("Load ERICA checkpoint")
        ckpt = torch.load("/st2/mkkang/ERICA/ERICA_bert_uncased_EP+RP")
        model.bert.load_state_dict(ckpt["bert-base"], strict=False)

        word_emb = model.bert.embeddings.word_embeddings.weight
        highlight_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)
        placeholder_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["#"])[0]].unsqueeze(0)
        marker_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["*"])[0]].unsqueeze(0)

        model.bert.embeddings.word_embeddings.weight.data = torch.cat(
            [word_emb, highlight_emb, placeholder_emb, marker_emb]
        )
    else:
        model = BertForEntitySpanQA.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        args.tokenizer = tokenizer
    
        word_emb = model.bert.embeddings.word_embeddings.weight
        highlight_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)
        placeholder_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["#"])[0]].unsqueeze(0)
        marker_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["*"])[0]].unsqueeze(0)

        model.bert.embeddings.word_embeddings.weight.data = torch.cat(
            [word_emb, highlight_emb, placeholder_emb, marker_emb]
        )
    model.config.vocab_size += 3
    tokenizer.add_special_tokens(
        dict(additional_special_tokens=[HIGHLIGHT_TOKEN, PLACEHOLDER_TOKEN, ENTITY_MARKER_TOKEN])
    )

    return model, tokenizer

def run(args):
    set_seed(args.seed)
    args.device = 'cuda'

    model, tokenizer = initialize_model(args)

    ### Original Code ###
    results = {}
    writer = Writer(args)
    
    # model = LukeForEntitySpanQA(args)
    # model.load_state_dict(args.model_weights, strict=False)
    model.to(args.device)

    train_dataloader, _, _, _ = load_examples(args, "train")

    num_train_steps_per_epoch = len(train_dataloader)
    num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)

    best_dev_score = [-1]
    best_weights = [None]

    def step_callback(model, global_step):
        if global_step % num_train_steps_per_epoch == 0 and args.local_rank in (0, -1):
            epoch = int(global_step / num_train_steps_per_epoch - 1)
            dev_results = evaluate(args, model, fold="dev")
            tqdm.write("dev: " + str(dev_results))
            results.update({f"dev_{k}_epoch{epoch}": v for k, v in dev_results.items()})
            writer.report_acc(dev_results["exact_match"], dev_results["f1"], epoch)
            if dev_results["exact_match"] > best_dev_score[0]:
                if hasattr(model, "module"):
                    best_weights[0] = {k: v.to("cpu").clone() for k, v in model.module.state_dict().items()}
                else:
                    best_weights[0] = {k: v.to("cpu").clone() for k, v in model.state_dict().items()}
                best_dev_score[0] = dev_results["exact_match"]
                results["best_epoch"] = epoch
            model.train()

    trainer = Trainer(
        args, model=model, 
        dataloader=train_dataloader, 
        num_train_steps=num_train_steps,
        writer=writer,
        step_callback=step_callback,
    )
    trainer.train()

    print(results)

    logger.info("Saving the model checkpoint to %s", args.output_dir)
    torch.save(best_weights[0], os.path.join(args.output_dir, WEIGHTS_NAME))
    # model.save_pretrained(args.output_dir)

    # Evaluate
    model, tokenizer = initialize_model(args)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, WEIGHTS_NAME), map_location="cpu"))
    model.to(args.device)

    output_file = os.path.join(args.output_dir, "predictions.json")
    results = evaluate(args, model, fold="test", output_file=output_file)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f)
    
    return results

def evaluate(args, model, fold="dev", output_file=None):
    dataloader, examples, features, processor = load_examples(args, fold)
    doc_predictions = defaultdict(list)
    for batch in tqdm(dataloader, desc="Eval"):
        model.eval()
        inputs = {k: v.to(args.device) for k, v in batch.items() if k != "feature_indices"}
        with torch.no_grad():
            logits = model(**inputs)

        for i, feature_index in enumerate(batch["feature_indices"]):
            feature = features[feature_index.item()]
            max_logit, max_index = logits[i].detach().max(dim=0)
            example_id = examples[feature.example_index].qas_id
            entity = feature.entities[max_index.item()]
            doc_predictions[example_id].append((max_logit, entity))

    predictions = {k: sorted(v, key=lambda o: o[0])[-1][1]["text"] for k, v in doc_predictions.items()}
    if output_file:
        with open(output_file, "w") as f:
            json.dump(predictions, f)

    if fold == "dev":
        with open(os.path.join(args.data_dir, processor.dev_file)) as f:
            dev_data = json.load(f)["data"]
    else:
        with open(os.path.join(args.data_dir, processor.test_file)) as f:
            dev_data = json.load(f)["data"]

    return evaluate_on_record(dev_data, predictions)[0]


def load_examples(args, fold):
    processor = RecordProcessor()
    if fold == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif fold == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    # bert_model_name = args.model_config.bert_model_name

    if args.lm_type == "roberta":
        segment_b_id = 0
        add_extra_sep_token = True
    else:
        segment_b_id = 1
        add_extra_sep_token = False

    # segment_b_id = 0
    # add_extra_sep_token = True

    logger.info("Creating features from the dataset...")

    if args.lm_type in ["roberta"]:
        pickle_name = "train_features.pkl"
    else:
        pickle_name = "train_features_bert.pkl"
    
    if args.read_data or (fold == "dev" or fold == "test"):
        features = convert_examples_to_features(
            examples,
            args.tokenizer,
            args.max_seq_length,
            args.max_mention_length,
            args.doc_stride,
            args.max_query_length,
            segment_b_id,
            add_extra_sep_token,
        )
        if fold == "train":
            with open(os.path.join(args.pickle_folder, pickle_name), 'wb+') as f:
                pickle.dump(features, f)
    else:
        with open(os.path.join(args.pickle_folder, pickle_name), 'rb') as f:
            features = pickle.load(f)

    def collate_fn(batch):
        def create_padded_sequence(target, padding_value):
            if isinstance(target, str):
                tensors = [torch.tensor(getattr(o[1], target), dtype=torch.long) for o in batch]
            else:
                tensors = [torch.tensor(o, dtype=torch.long) for o in target]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        entity_attention_mask = []
        entity_position_ids = []

        for _, item in batch:
            entity_length = len(item.entity_position_ids) + 1
            entity_attention_mask.append([1] * entity_length)
            # entity_position_ids.append(item.placeholder_position_ids + item.entity_position_ids)
            entity_position_id = []
            for position_ids in item.placeholder_position_ids + item.entity_position_ids:
                entity_position_id.append([min(a for a in position_ids if a > 0), max(position_ids)])
            entity_position_ids.append(entity_position_id)

            if entity_length == 1:
                # entity_position_ids[-1].append([-1] * args.max_mention_length)
                entity_position_ids[-1].append([-1, -1])
                entity_attention_mask[-1].append(0)

        ret = dict(
            input_ids=create_padded_sequence("word_ids", args.tokenizer.pad_token_id),
            attention_mask=create_padded_sequence("word_attention_mask", 0),
            token_type_ids=create_padded_sequence("word_segment_ids", 0),
            entity_attention_mask=create_padded_sequence(entity_attention_mask, 0),
            entity_position_ids=create_padded_sequence(entity_position_ids, -1),
        )
        if fold == "train":
            ret["labels"] = create_padded_sequence("labels", 0)
        else:
            ret["feature_indices"] = torch.tensor([o[0] for o in batch], dtype=torch.long)

        return ret

    if fold == "train":
        if args.local_rank == -1:
            sampler = RandomSampler(features)
        else:
            sampler = DistributedSampler(features)
        dataloader = DataLoader(
            list(enumerate(features)), sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn
        )
    else:
        dataloader = DataLoader(list(enumerate(features)), batch_size=args.eval_batch_size, collate_fn=collate_fn)

    return dataloader, examples, features, processor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dataset/ReCoRD_split")
    parser.add_argument("--train_file", type=str,
                        default="train.json")
    parser.add_argument("--dev_file", type=str,
                        default="dev.json")
    parser.add_argument("--pickle_folder", type=str, default="./dataset/ReCoRD_split")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--train_batch_size", type=int, default=12)
    parser.add_argument("--eval_batch_size", type=int, default=12)

    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_mention_length", type=int, default=30)
    parser.add_argument("--max_query_length", type=int, default=90)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")

    parser.add_argument("--model_dir", type=str, default="./save/no-name")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=1004)

    parser.add_argument("--version_2_with_negative", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./result")
    parser.add_argument("--n_best_size", type=int, default=20)
    parser.add_argument("--verbose_logging", action="store_true")
    parser.add_argument("--max_answer_length", type=int, default=30)
    parser.add_argument("--save_steps", type=int, default=0)

    parser.add_argument("--null_score_diff_threshold", type=float, default=0.0)
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--read_data", action="store_true",
                        help="read data from json file")
    # parser.add_argument("--electra", action="store_true")
    parser.add_argument("--lm_type", default="roberta",
                        choices=["bert", "electra", "electra-base", "roberta", "bert-erica"])

    args = parser.parse_args()

    args.pickle_folder = args.data_dir

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    run(args)