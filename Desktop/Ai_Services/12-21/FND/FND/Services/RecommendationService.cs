// Services/RecommendationService.cs
using System.Net.Http.Json;
using System.Text.Json;

public class RecommendationService
{
    private readonly HttpClient _httpClient;
    private readonly IConfiguration _configuration;

    public RecommendationService(HttpClient httpClient, IConfiguration configuration)
    {
        _httpClient = httpClient;
        _configuration = configuration;
    }

    public async Task<List<NewsRecommendation>> GetRecommendationsAsync(string userId, int topK = 5)
    {
        try
        {
            var pythonServiceUrl = _configuration["PythonService:Url"];
            var response = await _httpClient.PostAsJsonAsync(
                $"{pythonServiceUrl}/recommend",
                new { user_id = userId, top_k = topK });

            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<RecommendationResult>();
            return result?.Recommendations ?? new List<NewsRecommendation>();
        }
        catch (Exception ex)
        {
            // Log error
            Console.WriteLine($"Error getting recommendations: {ex.Message}");
            return new List<NewsRecommendation>();
        }
    }
}

public record RecommendationResult(List<NewsRecommendation> Recommendations);
public record NewsRecommendation(string NewsId, string Title, string Abstract, string Category, string Url);