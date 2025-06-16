using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;

public class RagService
{
    private readonly HttpClient _httpClient;

    public RagService(HttpClient httpClient)
    {
        _httpClient = httpClient;
        _httpClient.BaseAddress = new Uri("http://localhost:8501/");
    }

    public async Task<string> GetAnswerFromRAG(string question)
    {
        var request = new { question = question };
        var response = await _httpClient.PostAsJsonAsync("rag_answer", request);
        if (response.IsSuccessStatusCode)
        {
            var result = await response.Content.ReadFromJsonAsync<RagResponse>();
            return result?.Answer ?? "لا توجد إجابة.";
        }
        return "حدث خطأ.";
    }

    private class RagResponse
    {
        public string Answer { get; set; }
    }
}
