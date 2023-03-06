import torch, torchvision
from transformers import AutoModelForSequenceClassification

def main():
    model = torchvision.models.vgg16(pretrained=True)

    #model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=10)

    #model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased') 

    model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2') 

    g = torch.Tensor()
    for params in model.parameters():
        t = params.data
        t = torch.flatten(t)
        g = torch.cat((g,t))
    
    
    print("Mean = ", end="")
    print(torch.mean(g).item())
    print("STD = ", end="")
    print(torch.std(g).item())
    print("Median = ", end="")
    print(torch.median(g).item())
    print("Max = ", end="")
    print(torch.max(g).item())
    print("Min = ", end="")
    print(torch.min(g).item())
    print()
        


if __name__=="__main__":
    main()
