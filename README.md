# VGLC_LLM_Finetunning

Dirs:

**TheVGLC:** Diretorio com as fases do VGLC. Algumas fases do Super Mario Bros 2 sofreram preprocessamento

**preprocessing:** Notebooks que pegam as fases em txt e transformam no formato necessário para finetunning

**level_json:** Fases no formato de JSON

**huggingface:** Notebook que uso para salvar o dataset no huggingface

No root tem os notebooks para finetunning. Para usar corretamente, preencha um .env no seu root com as variaveis que estão no .env.example