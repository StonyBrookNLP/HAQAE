# functions used shared between the generate methods.


def transform(output, dict):
    out = ""
    for i in output:
        out += " " + dict[i]
    return out


def get_pred_events(outputs, vocab, number=4):
    predicted_events = []
    for output in outputs[0]:
        predicted = transform(output, vocab.itos)
        # TODO explicitly cut after 4th word as <TUP> is not generated
        #predicted = predicted.split("<TUP>")[1].strip().split()[:number] # <TUP> ...
        #predicted = " ".join(predicted)
        predicted_events.append(predicted)

    predicted_events_str = " || ".join(predicted_events)
    return predicted_events_str


# tuple = (verb, arg11, arg2, prep)
def get_tups(events): 
    return [tuple(each.strip().split(" ")) for each in events] # list of tups  




