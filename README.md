# Sistemas Evolutivos Aplicados à Robótica — Geração de Imagens com Stable Diffusion

## Sobre o Projeto

Este projeto apresenta um estudo utilizando a biblioteca **Diffusers**, com foco em tarefas de **geração de imagens por Inteligência Artificial** utilizando técnicas de **Deep Learning** e modelos baseados em difusão.

O objetivo principal consiste em analisar a capacidade do modelo em gerar automaticamente imagens a partir de descrições textuais (*prompts*), verificando a qualidade visual dos resultados obtidos e sua correspondência com as descrições fornecidas.

Foram realizados experimentos utilizando o modelo **runwayml/stable-diffusion-v1-5**, permitindo a criação de diferentes tipos de imagens envolvendo cenários urbanos, paisagens naturais, animais, elementos espaciais e objetos de fantasia.

---

## Tecnologias Utilizadas

As seguintes tecnologias foram utilizadas durante o desenvolvimento do projeto:

- Python
- Diffusers
- PyTorch
- Transformers
- Stable Diffusion
- Google Colab
- Inteligência Artificial
- Deep Learning

---

## Recursos Utilizados

Os seguintes recursos disponibilizados durante os experimentos foram utilizados:

| Recurso | Função |
|----------|---------|
| Stable Diffusion v1.5 | Geração de imagens |
| Prompt Processing | Interpretação do texto |
| GPU Acceleration | Aceleração do processamento |
| Diffusers | Implementação do modelo |
| Image Generation | Criação automática das imagens |

---

## Metodologia

Os experimentos foram realizados seguindo as seguintes etapas:

1. Instalação automática das bibliotecas necessárias;
2. Verificação da disponibilidade de GPU;
3. Carregamento do modelo `runwayml/stable-diffusion-v1-5`;
4. Definição dos prompts de entrada;
5. Geração automática das imagens;
6. Medição do tempo de processamento;
7. Análise dos resultados obtidos.

Foram utilizados seis prompts distintos para avaliar a capacidade do modelo em interpretar diferentes tipos de cenários e objetos.

---

## Prompts Utilizados

| Imagem | Prompt |
|----------|---------|
| Imagem 1 | A futuristic city at night with neon lights |
| Imagem 2 | A medieval castle on top of a mountain |
| Imagem 3 | A realistic cat wearing sunglasses |
| Imagem 4 | A tropical beach during sunset |
| Imagem 5 | An astronaut walking on Mars |
| Imagem 6 | A dragon flying over a forest |

---

## Resultados Obtidos

### Tempos de geração

| Imagem | Tempo |
|----------|---------|
| Imagem 1 | 1230.86 segundos |
| Imagem 2 | 1198.89 segundos |
| Imagem 3 | 1193.32 segundos |
| Imagem 4 | 1185.74 segundos |
| Imagem 5 | 1198.82 segundos |
| Imagem 6 | 1193.15 segundos |

**Tempo médio aproximado:** 1200.13 segundos

---

## Análise dos Resultados

### Imagem 1 — Cidade futurista

**Prompt:** `A futuristic city at night with neon lights`

O modelo apresentou boa capacidade de gerar cenários urbanos complexos, produzindo elementos tecnológicos e iluminação neon compatíveis com a descrição fornecida.

---

### Imagem 2 — Castelo medieval

**Prompt:** `A medieval castle on top of a mountain`

A imagem apresentou elementos arquitetônicos característicos de construções medievais, incluindo torres e muralhas, mantendo coerência com o ambiente montanhoso.

---

### Imagem 3 — Gato utilizando óculos

**Prompt:** `A realistic cat wearing sunglasses`

O modelo conseguiu representar corretamente o animal solicitado, embora pequenos detalhes dos acessórios possam apresentar leves distorções.

---

### Imagem 4 — Praia tropical

**Prompt:** `A tropical beach during sunset`

A imagem gerada apresentou elevado nível de fidelidade ao prompt, contendo mar, areia, vegetação e iluminação característica do pôr do sol.

---

### Imagem 5 — Astronauta em Marte

**Prompt:** `An astronaut walking on Mars`

Foram observadas características adequadas do ambiente marciano, juntamente com elementos associados ao traje espacial.

---

### Imagem 6 — Dragão sobre floresta

**Prompt:** `A dragon flying over a forest`

O modelo demonstrou boa capacidade para criação de elementos de fantasia, mantendo coerência entre o personagem principal e o cenário.

---

## Conclusão

Os resultados obtidos demonstraram que o modelo **runwayml/stable-diffusion-v1-5** apresentou desempenho satisfatório na geração automática de imagens a partir de descrições textuais.

Foi possível observar boa correspondência entre os prompts utilizados e as imagens geradas. Cenários naturais e urbanos apresentaram resultados mais detalhados, enquanto elementos específicos podem apresentar pequenas inconsistências em detalhes mais complexos.

Os experimentos também demonstraram a aplicação prática de técnicas de inteligência artificial e aprendizado profundo em tarefas de geração automática de conteúdo visual.

## Código disponível em:

```python
!pip -q install diffusers transformers accelerate safetensors

from diffusers import StableDiffusionPipeline
import torch
import time
from IPython.display import display

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
)

pipe = pipe.to(device)

prompts = [
    "A futuristic city at night with neon lights",
    "A medieval castle on top of a mountain",
    "A realistic cat wearing sunglasses",
    "A tropical beach during sunset",
    "An astronaut walking on Mars",
    "A dragon flying over a forest"
]

for i, prompt in enumerate(prompts):

    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    image.save(f"imagem_{i+1}.png")

    display(image)
```
