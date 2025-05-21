## Rodando avaliações

#### 1. inference_batch
##### **Input**

###### **AVISO**: Rodar os modelos com path e sem path **SEPARADAMENTE**

Dentro do `inference/specific_inference/inference_batch.ipynb` tem um notebook para você rodar os seus modelos treinados.
```
models = {
    "llama-3": [
        "/home/pressprexx/Code/AKCITGaming/Paper_LLM_PCG_Geral/VGLC_LLM_Finetunning/models/mario/Llama-3.1-8B-Instruct-unsloth-bnb-4bit-mario-teste1"
    ],
    "qwen-3": [
        "/home/pressprexx/Code/AKCITGaming/Paper_LLM_PCG_Geral/VGLC_LLM_Finetunning/models/mario/Qwen3-14B-Instruct-bnb-4bit-mario-horizontal-newline-teste1"
    ],
    "qwen-2.5": [
        "/home/pressprexx/Code/AKCITGaming/Paper_LLM_PCG_Geral/VGLC_LLM_Finetunning/models/mario/Qwen-2.5-14b-horizontal-newline-1epoch-mario-teste1"
    ],
    "gemma-3": [
        "/home/pressprexx/Code/AKCITGaming/Paper_LLM_PCG_Geral/VGLC_LLM_Finetunning/models/mario/gemma-3-12b-it-unsloth-bnb-4bit-mariogpt-teste1"
    ]
}

temperatures = [0.7, 1.0, 1.2, 1.5, 2.0]

num_of_samples = 5

game_type = "mario"  # options: "mario", "loderunner", "kidicarus", "rainbowisland"
```

Você só precisa mudar os params acima, onde:
- **models:** O caminnho dos modelos que você treinou, separado por familia de modelo. Se você não treinou um, é só retirar. Exemplo:
```
models = {
    "llama-3": [
        "/home/pressprexx/Code/AKCITGaming/Paper_LLM_PCG_Geral/VGLC_LLM_Finetunning/models/mario/Llama-3.1-8B-Instruct-unsloth-bnb-4bit-mario-teste1"
    ],
    "qwen-2.5": [
        "/home/pressprexx/Code/AKCITGaming/Paper_LLM_PCG_Geral/VGLC_LLM_Finetunning/models/mario/Qwen-2.5-14b-horizontal-newline-1epoch-mario-teste1"
    ]
}
```
Lembre de não misturar modelos com e sem path. Modelos com separadores diferentes não tem problema misturar

- **game_type:** O jogo que você usou para o treino. Só pode ter um, copie e cole do comentário na linha o jogo que você quer jogar.

<br>

##### **Output**

O output será um json e um pdf com o nome `level_generation_results_{data}.json` e `level_generation_results_{data}.pdf`


#### 2. metrics_batch
##### **Input**

O input será o JSON gerado pelo `inference_batch` junto com o tipo de jogo e se o modelo usado for com caminho. Como o anterior, não pode misturar modelos com e sem path.

```
# params
input_json_paths = [
    "level_generation_results_20250521_100847.json",
    "level_generation_results_20250521_100328.json"
    # Add more file paths as needed
]

game_type = "mario" # mario, kid_icarus, lode_runner, rainbow_island
path = False
```

- **input_json_paths:** JSONs dos resultados obtidos no JSON. Pode misturar familias de modelos diferentes (qwen, llama...)
- **game_type:** O jogo a ser avaliado. Usar as opções do comentário, são diferentes das opções do `inference_batch`
- **path:** Se os modelos são com caminho ou não. False para sem caminho, True para com caminho.