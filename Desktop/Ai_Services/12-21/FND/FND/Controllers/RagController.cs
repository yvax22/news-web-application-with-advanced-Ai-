using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Http;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json;
using FND.Models;
using FND.ViewModel;
using System.Text;
using System.Text.Json;
using System.Net.Http;
using System.Collections.Generic;

public class RagController : Controller
{
    private readonly RagService _ragService;
    private readonly Entity _db;
    private readonly IHttpClientFactory _httpClientFactory;

    public RagController(RagService ragService, Entity db, IHttpClientFactory httpClientFactory)
    {
        _ragService = ragService;
        _db = db;
        _httpClientFactory = httpClientFactory;
    }

    [HttpGet]
    public IActionResult Chat()
    {
        var model = new ChatViewModel
        {
            Messages = new List<ChatMessage>()
        };
        return View("Chat", model);
    }

    [HttpPost]
    public async Task<IActionResult> Ask(ChatViewModel model)
    {
        if (model.Messages == null)
            model.Messages = new List<ChatMessage>();

        try
        {
            var client = _httpClientFactory.CreateClient();
            var requestJson = System.Text.Json.JsonSerializer.Serialize(new { question = model.Question });

            var response = await client.PostAsync(
                "http://localhost:8500/rag_answer",
                new StringContent(requestJson, Encoding.UTF8, "application/json")
            );

            var responseContent = await response.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(responseContent);

            // ✅ استخراج الإجابة بأمان
            string answer = "";
            try
            {
                answer = doc.RootElement.GetProperty("answer").GetString();
            }
            catch
            {
                answer = " لم يتم توليد إجابة من النموذج.";
            }

          

            // ✅ تركيب الرد النهائي
            var finalAnswer = answer;

            

            

            // ✅ حفظ الرسائل في المحادثة
            model.Messages.Add(new ChatMessage { Role = "user", Message = model.Question });
            model.Messages.Add(new ChatMessage { Role = "bot", Message = finalAnswer });
            model.Question = "";

            // ✅ حفظ في قاعدة البيانات
            var chat = new ChatHistory
            {
                UserId = User.Identity?.Name ?? "Guest",
                Question = model.Messages.FirstOrDefault(m => m.Role == "user")?.Message,
                Answer = model.Messages.LastOrDefault(m => m.Role == "bot")?.Message,
                CreatedAt = DateTime.Now
            };

            _db.ChatHistories.Add(chat);
            await _db.SaveChangesAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine(" خطأ أثناء المعالجة: " + ex.Message);
            model.Messages.Add(new ChatMessage { Role = "bot", Message = " خطأ: {ex.Message}" });
        }

        return View("Chat", model);
    }



    [HttpPost]
    public async Task<IActionResult> ChatApi([FromBody] RagQuestionModel model)
    {
        string userId = User.Identity?.Name ?? "Guest";
        string answer = await _ragService.GetAnswerFromRAG(model.Question);

        var chat = new ChatHistory
        {
            UserId = userId,
            Question = model.Question,
            Answer = answer,
            CreatedAt = DateTime.Now
        };

        _db.ChatHistories.Add(chat);
        await _db.SaveChangesAsync();

        return Json(new { answer });
    }

    public class RagQuestionModel
    {
        public string Question { get; set; }
    }

    [HttpGet]
    public IActionResult History()
    {
        string userId = User.Identity?.Name ?? "Guest";
        var chats = _db.ChatHistories
            .Where(c => c.UserId == userId)
            .OrderByDescending(c => c.CreatedAt)
            .ToList();

        var model = new ChatViewModel
        {
            Messages = chats.Select(c => new ChatMessage
            {
                Role = "user",
                Message = $" {c.Question}\n {c.Answer}"
            }).ToList()
        };

        return View("Chat", model); // ✅ تمرير ViewModel
    }
    public IActionResult Summarize()
    {
        return View();
    }
}
