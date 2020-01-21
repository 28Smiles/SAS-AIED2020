from .dataset import load_semeval, tokenize, dataset
from torch.utils.data import DataLoader


def val_dataset(name, type, lang, tokenizer, sequence_length):
    return DataLoader(dataset(tokenize(load_semeval(name, type, lang), tokenizer, sequence_length)), batch_size=32)


def val_datasets(tokenizer, sequence_length):
    return {
        'union_unseen_answers_en': val_dataset('union', 'unseen_answers', 'en', tokenizer, sequence_length),
        'union_unseen_questions_en': val_dataset('union', 'unseen_questions', 'en', tokenizer, sequence_length),
        'union_unseen_domains_en': val_dataset('union', 'unseen_domains', 'en', tokenizer, sequence_length),
        'union_unseen_answers_de': val_dataset('union', 'unseen_answers', 'de', tokenizer, sequence_length),
        'union_unseen_questions_de': val_dataset('union', 'unseen_questions', 'de', tokenizer, sequence_length),
        'union_unseen_domains_de': val_dataset('union', 'unseen_domains', 'de', tokenizer, sequence_length),

        'beetle_unseen_answers_en': val_dataset('beetle', 'unseen_answers', 'en', tokenizer, sequence_length),
        'beetle_unseen_questions_en': val_dataset('beetle', 'unseen_questions', 'en', tokenizer, sequence_length),
        'beetle_unseen_answers_de': val_dataset('beetle', 'unseen_answers', 'de', tokenizer, sequence_length),
        'beetle_unseen_questions_de': val_dataset('beetle', 'unseen_questions', 'de', tokenizer, sequence_length),

        'sciEntsBank_unseen_answers_en': val_dataset('sciEntsBank', 'unseen_answers', 'en', tokenizer, sequence_length),
        'sciEntsBank_unseen_questions_en': val_dataset('sciEntsBank', 'unseen_questions', 'en', tokenizer,
                                                       sequence_length),
        'sciEntsBank_unseen_domains_en': val_dataset('sciEntsBank', 'unseen_domains', 'en', tokenizer, sequence_length),
        'sciEntsBank_unseen_answers_de': val_dataset('sciEntsBank', 'unseen_answers', 'de', tokenizer, sequence_length),
        'sciEntsBank_unseen_questions_de': val_dataset('sciEntsBank', 'unseen_questions', 'de', tokenizer,
                                                       sequence_length),
        'sciEntsBank_unseen_domains_de': val_dataset('sciEntsBank', 'unseen_domains', 'de', tokenizer, sequence_length),
    }
