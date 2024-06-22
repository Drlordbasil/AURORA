# test_lobes.py

from Brain_modules.lobes.lobes_processing import LobesProcessing

def combine_thoughts(thoughts):
    combined_thought = "\n".join(f"{lobe}: {thought}" for lobe, thought in thoughts.items())
    return combined_thought

def test_lobes():
    lobes_processor = LobesProcessing(image_vision=False)
    test_prompts = [
        "The box is above the table, near the window",

        "Process this sentence without any spatial or numerical content",
        "Error: setting an array element with a sequence",
        "sup man its me Anthony"
    ]
    
    for prompt in test_prompts:
        print(f"\nTesting with prompt: '{prompt}'")
        thoughts = {}
        for lobe_name in lobes_processor.lobes.keys():
            result = lobes_processor.process_lobe(lobe_name, prompt)
            thoughts[lobe_name] = result
            print(f"{lobe_name.capitalize()} Lobe: {result}")

        final_thought = combine_thoughts(thoughts)
        print(f"\nFinal Combined Thought for prompt '{prompt}':\n{final_thought}\n")

if __name__ == "__main__":
    test_lobes()
