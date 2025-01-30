from nltk import word_tokenize
def add_personAB(dialog_context, max_words = 100000, orders = ['Person A: ', 'Person B: ']):
    if type(dialog_context) != list:
        dialog_context = dialog_context.split('\n')
    context = list(reversed(dialog_context))
    
    for i in range(0,len(context)):
        if i % 2 == 0:
            context[i] = orders[0] + context[i]
        else:
            context[i] = orders[1] + context[i]
        
        tokens = word_tokenize('\n'.join(context[:i+1]))
        if len(tokens) > max_words:
            break
    
    if len(tokens) < max_words:
        context = context[:i+1]
    else:
        context = context[:i]
    
    context = list(reversed(context))
    return '\n'.join(context)

def dialog_prompt(context):
    prompt  = "Given this conversation:\n\n"
    prompt += add_personAB(context) + "\n\n"
    prompt += "Imagine you are person B and act as if you were a real individual. Please write the next response for person B.\n"
    prompt += "Person B:"
    return prompt
