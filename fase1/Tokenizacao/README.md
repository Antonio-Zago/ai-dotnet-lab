# Tokenização: caractere, palavra e BPE

Fase 1 · Semana 1 do plano de estudos.

Implementação do zero dos três tipos de tokenização, comparados entre si e com o tokenizador real do GPT-4o (`o200k_base`, ~200 mil tokens).

| Tokenizador | Ideia | Ponto forte | Ponto fraco |
|---|---|---|---|
| Caractere | 1 caractere = 1 token | Vocabulário minúsculo | Sequências longas; caractere não visto vira `<UNK>` |
| Palavra | 1 palavra = 1 token | Sequências curtas | Palavra nova vira `<UNK>`; perde maiúsculas e espaços |
| BPE (byte-level) | Começa nos 256 bytes e junta o par mais frequente até o tamanho escolhido | Nunca tem `<UNK>`; ida e volta exata | Tokens isolados podem ser meio caractere UTF-8 (`�`) |

## Como rodar

```bash
dotnet run
```

Requer .NET 9 ou 10.

## O que observar

- A frase 1 só usa palavras do corpus; a frase 2 tem palavras novas ("refatorou", "escreveu", "desenvolvedora") e um emoji.
- Compare a contagem de tokens e a coluna `ida-e-volta` de cada tokenizador.
- Os espaços aparecem como `·`, grudados no início das palavras (efeito da pré-tokenização).

## Experimentos

- [ ] Mudar `vocabSize: 256 + 300` para `256 + 20` e `256 + 1000` e ver a contagem de tokens do BPE mudar
- [ ] Trocar o corpus por um texto em inglês e tokenizar a mesma frase em português
- [ ] Contar tokens de um arquivo C# e de um JSON reais com o tokenizador do GPT-4o
