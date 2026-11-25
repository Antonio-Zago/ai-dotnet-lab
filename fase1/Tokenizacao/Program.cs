// =============================================================================
//  Tokenização na prática: caractere, palavra e BPE (byte-level) lado a lado,
//  comparados com o tokenizador real do GPT-4o.
//
//  Como rodar (nesta pasta):  dotnet run   — requer .NET 9 ou 10
// =============================================================================

using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

Console.OutputEncoding = Encoding.UTF8;

// Corpus de treino: pequeno de propósito, para dar para entender o que cada
// tokenizador aprende. Num LLM real, seriam centenas de GB de texto.
const string Corpus = """
    O desenvolvedor abriu o código e encontrou um bug na função de autenticação.
    O bug estava na validação do token: a função não conferia a data de expiração.
    O desenvolvedor corrigiu a função, rodou os testes e publicou a correção.
    A equipe revisou o código da correção e aprovou a mudança na autenticação.
    Depois, o desenvolvedor documentou a função e explicou o bug para a equipe.
    A validação do token agora confere a data, a assinatura e o emissor.
    Os testes de autenticação passaram e a equipe publicou a nova versão do código.
    """;

ITokenizer[] tokenizadores =
[
    new CharTokenizer(Corpus),
    new WordTokenizer(Corpus, maxVocab: 500),
    new BpeTokenizer(Corpus, vocabSize: 256 + 300),
];

// Tokenizador real usado pelo GPT-4o (byte-level BPE, vocabulário ~200 mil)
var gpt4o = TiktokenTokenizer.CreateForModel("gpt-4o");

string[] testes =
[
    // Só palavras que aparecem no corpus
    "O desenvolvedor corrigiu o bug na função de autenticação.",
    // Palavras novas ("desenvolvedora", "refatorou", "escreveu"), maiúscula nova e emoji
    "A desenvolvedora refatorou o código e escreveu testes 🚀",
];

foreach (var texto in testes)
{
    Console.WriteLine(new string('=', 90));
    Console.WriteLine($"Texto: \"{texto}\"  ({texto.Length} caracteres UTF-16)");
    Console.WriteLine();

    foreach (var tk in tokenizadores)
    {
        var ids = tk.Encode(texto);
        var volta = tk.Decode(ids);

        Console.WriteLine($"[{tk.Name}] vocabulário={tk.VocabSize}  tokens={ids.Length}  " +
                          $"ida-e-volta={(volta == texto ? "OK" : "PERDEU INFORMAÇÃO")}");
        Console.WriteLine("  Peças: " + string.Join("|", ids.Select(tk.TokenToString).Select(Visivel)));
        Console.WriteLine("  IDs:   " + string.Join(", ", ids.Take(20)) + (ids.Length > 20 ? ", ..." : ""));
        if (volta != texto)
            Console.WriteLine($"  Volta: \"{volta}\"");
        Console.WriteLine();
    }

    var reais = gpt4o.EncodeToTokens(texto, out _);
    Console.WriteLine($"[GPT-4o real] vocabulário≈200 mil  tokens={reais.Count}");
    Console.WriteLine("  Peças: " + string.Join("|", reais.Select(t => Visivel(t.Value))));
    Console.WriteLine("  IDs:   " + string.Join(", ", reais.Select(t => t.Id)));
    Console.WriteLine();
}

// Mostra espaços como "·" para ficar claro que eles fazem parte do token
static string Visivel(string s) => s.Replace(" ", "·").Replace("\n", "\\n");


// =============================================================================
//  Contrato comum
// =============================================================================
interface ITokenizer
{
    string Name { get; }
    int VocabSize { get; }
    int[] Encode(string text);
    string Decode(int[] ids);
    string TokenToString(int id);
}


// =============================================================================
//  1. CHARACTER-LEVEL: cada caractere é um token
//     - Vocabulário minúsculo (só os caracteres vistos no corpus)
//     - Sequências longas: 1 token por caractere
//     - Caractere nunca visto (ex.: emoji) vira <UNK>
// =============================================================================
sealed class CharTokenizer : ITokenizer
{
    private const string Unk = "<UNK>";
    private readonly List<string> _itos = [Unk];                         // id → caractere
    private readonly Dictionary<string, int> _stoi = new() { [Unk] = 0 }; // caractere → id

    public CharTokenizer(string corpus)
    {
        // EnumerateRunes trata emoji e outros caracteres fora do BMP como 1 caractere só
        var caracteres = corpus.EnumerateRunes()
                               .Select(r => r.ToString())
                               .Distinct()
                               .Order(StringComparer.Ordinal);

        foreach (var c in caracteres)
        {
            _stoi[c] = _itos.Count;
            _itos.Add(c);
        }
    }

    public string Name => "Caractere";
    public int VocabSize => _itos.Count;

    //Tenta encontrar o caracter no dicionário de caractéres - números inteiros que foi feito no treinamento
    public int[] Encode(string text) =>
        text.EnumerateRunes()
            .Select(r => _stoi.GetValueOrDefault(r.ToString(), 0))
            .ToArray();

    //Faz o contrário e converte os ids de volta na palavra
    public string Decode(int[] ids) => string.Concat(ids.Select(TokenToString));

    public string TokenToString(int id) => _itos[id];
}


// =============================================================================
//  2. WORD-LEVEL: cada palavra (ou sinal de pontuação) é um token
//     - Sequências curtas
//     - Vocabulário limitado às N palavras mais frequentes
//     - Palavra nunca vista vira <UNK>; "desenvolvedor" e "desenvolvedora" não têm relação
//     - Perde maiúsculas e espaçamento original (não dá para reconstruir o texto exato)
// =============================================================================
sealed class WordTokenizer : ITokenizer
{
    private static readonly Regex Splitter = new(@"\w+|[^\w\s]", RegexOptions.Compiled);

    private const string Unk = "<UNK>";
    private readonly List<string> _itos = [Unk];
    private readonly Dictionary<string, int> _stoi = new() { [Unk] = 0 };

    public WordTokenizer(string corpus, int maxVocab)
    {
        var maisFrequentes = Split(corpus)
            .GroupBy(w => w)
            .OrderByDescending(g => g.Count())
            .ThenBy(g => g.Key, StringComparer.Ordinal)
            .Take(maxVocab - 1)                  // -1 porque <UNK> já ocupa uma vaga
            .Select(g => g.Key);

        foreach (var w in maisFrequentes)
        {
            _stoi[w] = _itos.Count;
            _itos.Add(w);
        }
    }

    private static IEnumerable<string> Split(string text) =>
        Splitter.Matches(text.ToLowerInvariant()).Select(m => m.Value);

    public string Name => "Palavra";
    public int VocabSize => _itos.Count;

    public int[] Encode(string text) =>
        Split(text).Select(w => _stoi.GetValueOrDefault(w, 0)).ToArray();

    // Reconstrução aproximada: junta com espaços (o original não é recuperável)
    public string Decode(int[] ids) => string.Join(" ", ids.Select(TokenToString));

    public string TokenToString(int id) => _itos[id];
}


// =============================================================================
//  3. BYTE-LEVEL BPE: começa com os 256 bytes e junta o par mais frequente,
//     repetidamente, até atingir o tamanho de vocabulário escolhido.
//     - Nunca tem <UNK>: qualquer texto vira bytes
//     - Palavras frequentes viram 1 token; palavras novas são montadas com pedaços
//     - Ida e volta sempre exata
// =============================================================================
sealed class BpeTokenizer : ITokenizer
{
    // Pré-tokenização simplificada no estilo GPT: os merges nunca atravessam estes blocos.
    // Blocos: palavra com o espaço da frente | número | pontuação/símbolos | espaços
    private static readonly Regex PreTokenizer = new(
        @" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    // Par → id do token criado. Como os ids crescem na ordem do treino,
    // id menor = merge aprendido antes = maior prioridade no encode.
    private readonly Dictionary<(int, int), int> _merges = new();
    private readonly Dictionary<int, byte[]> _vocab = new();   // id → bytes que ele representa

    public BpeTokenizer(string corpus, int vocabSize)
    {
        for (int b = 0; b < 256; b++)
            _vocab[b] = [(byte)b];

        // Cada bloco único do corpus vira uma lista de bytes + quantas vezes aparece
        var blocos = PreTokenizer.Matches(corpus)
            .GroupBy(m => m.Value)
            .Select(g => (Ids: Encoding.UTF8.GetBytes(g.Key).Select(b => (int)b).ToList(),
                          Freq: g.Count()))
            .ToList();

        for (int novoId = 256; novoId < vocabSize; novoId++)   // ← "quando parar": vocabSize
        {
            // 1. Conta todos os pares vizinhos (ponderado pela frequência do bloco)
            var contagem = new Dictionary<(int, int), int>();
            foreach (var (ids, freq) in blocos)
                for (int i = 0; i < ids.Count - 1; i++)
                {
                    var par = (ids[i], ids[i + 1]);
                    contagem[par] = contagem.GetValueOrDefault(par) + freq;
                }

            if (contagem.Count == 0) break;

            // 2. Escolhe o par mais frequente
            var (melhor, vezes) = contagem.MaxBy(kv => kv.Value);
            if (vezes < 2) break;   // corpus pequeno: nada mais que valha a pena juntar

            // 3. Cria o token novo e substitui o par em todo o corpus
            _merges[melhor] = novoId;
            _vocab[novoId] = [.. _vocab[melhor.Item1], .. _vocab[melhor.Item2]];

            for (int b = 0; b < blocos.Count; b++)
                blocos[b] = (Merge(blocos[b].Ids, melhor, novoId), blocos[b].Freq);
        }
    }

    public string Name => "BPE (byte-level)";
    public int VocabSize => _vocab.Count;

    public int[] Encode(string text)
    {
        var resultado = new List<int>();

        foreach (Match bloco in PreTokenizer.Matches(text))
        {
            var ids = Encoding.UTF8.GetBytes(bloco.Value).Select(b => (int)b).ToList();

            // Aplica os merges na ordem em que foram aprendidos:
            // entre os pares presentes, junta sempre o de menor id (aprendido primeiro)
            while (ids.Count >= 2)
            {
                (int, int)? alvo = null;
                int menorId = int.MaxValue;

                for (int i = 0; i < ids.Count - 1; i++)
                {
                    var par = (ids[i], ids[i + 1]);
                    if (_merges.TryGetValue(par, out var id) && id < menorId)
                    {
                        menorId = id;
                        alvo = par;
                    }
                }

                if (alvo is null) break;   // nenhum merge se aplica mais
                ids = Merge(ids, alvo.Value, menorId);
            }

            resultado.AddRange(ids);
        }

        return resultado.ToArray();
    }

    public string Decode(int[] ids) =>
        Encoding.UTF8.GetString(ids.SelectMany(id => _vocab[id]).ToArray());

    // Um token isolado pode ser só metade de um caractere UTF-8 (ex.: metade do "ã");
    // nesse caso o .NET mostra "�". É esperado.
    public string TokenToString(int id) => Encoding.UTF8.GetString(_vocab[id]);

    private static List<int> Merge(List<int> ids, (int, int) par, int novoId)
    {
        var resultado = new List<int>(ids.Count);
        for (int i = 0; i < ids.Count; i++)
        {
            if (i < ids.Count - 1 && ids[i] == par.Item1 && ids[i + 1] == par.Item2)
            {
                resultado.Add(novoId);
                i++; // pula o segundo elemento do par
            }
            else
            {
                resultado.Add(ids[i]);
            }
        }
        return resultado;
    }
}
