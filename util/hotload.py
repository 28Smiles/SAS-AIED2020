def dynamic_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def load_model(prefix):
    return (
        dynamic_import('transformers.{}Tokenizer'.format(prefix)),
        dynamic_import('transformers.{}Config'.format(prefix)),
        dynamic_import('transformers.{}ForSequenceClassification'.format(prefix)),
    )
