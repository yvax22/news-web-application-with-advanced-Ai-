using FND.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Security.Claims;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.ServiceModel.Syndication;
using System.Xml;
using System.Text.RegularExpressions;
using System.Text.Json;
using System.Xml.Linq;
using HtmlAgilityPack;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
    
namespace FND.Controllers
{
    [Authorize]
    public class NewsController : Controller
    {
        private readonly Entity context;
        private readonly ILogger<NewsController> _logger;
        private readonly UserManager<ApplicationUser> userManager;
        private readonly HttpClient _httpClient;

        public NewsController(UserManager<ApplicationUser> userManager, Entity context, ILogger<NewsController> logger)
        {
            this.userManager = userManager;
            this.context = context;
            this._logger = logger;
            _httpClient = new HttpClient();

        }
      

        //...............................................................................................................new
        [Authorize(Roles = "Admin , Publisher")]
        public IActionResult New()
        {
            ViewBag.cats = context.Categories.ToList();
            ViewBag.images = Directory.GetFiles(Path.Combine(Directory.GetCurrentDirectory(), "wwwroot/images"))
                                      .Select(Path.GetFileName)
                                      .ToList();
            return View(new News());
        }

        [Authorize]
        [HttpPost]
        [Authorize]
        [HttpPost]
        public async Task<IActionResult> New(News model)
        {
            model.CreatedAt = DateTime.Now;
            model.UpdatedAt = DateTime.Now;
            model.IsFake = false;

            context.News.Add(model);
            context.SaveChanges();

            await SummarizeNewsAsync(model);

            //  إرسال ID الخبر إلى FastAPI لإضافته إلى FAISS
            var client = new HttpClient();
            var content = new FormUrlEncodedContent(new[]
            {
        new KeyValuePair<string, string>("news_id", model.Id.ToString())
          });

            try
            {
                var response = await client.PostAsync("http://localhost:8500/add_news_to_faiss", content);
                if (response.IsSuccessStatusCode)
                {
                    Console.WriteLine("✅ تم إرسال الخبر إلى FAISS بنجاح.");
                }
                else
                {
                    Console.WriteLine("❌ فشل في إرسال الخبر إلى FAISS.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("❌ خطأ أثناء الاتصال بـ FastAPI: " + ex.Message);
            }

            return RedirectToAction("New");
        }

        //...............................................................................................................AllNews
        public IActionResult AllNews(int page = 1, int pageSize = 21)
        {
            if (context.News == null)
                return Problem("News set is null.");

            var totalCount = context.News.Count();
            var totalPages = (int)Math.Ceiling((double)totalCount / pageSize);

            ViewBag.Page = page;
            ViewBag.PageSize = pageSize;
            ViewBag.TotalPages = totalPages;

            var items = context.News
                .OrderByDescending(n => n.CreatedAt)
                .Skip((page - 1) * pageSize)
                .Take(pageSize)
                .ToList();

            var likesCount = context.Likes
            .AsEnumerable()
            .GroupBy(l => l.NewsId)
            .ToDictionary(g => g.Key, g => g.Count());

            ViewBag.LikesCount = likesCount ?? new Dictionary<int, int>();
            ViewBag.TopLiked = context.News
            .OrderByDescending(n => n.Likes.Count)
            .Take(5)
            .ToList();

            var commentsDict = context.Comments
            .GroupBy(c => c.NewsId)
            .Select(g => new { NewsId = g.Key, Count = g.Count() })
            .ToDictionary(g => g.NewsId, g => g.Count);

            ViewBag.CommentsDict = commentsDict;


            return View(items);
        }






        //.................................................................................................................Search
        public IActionResult SearchResult(string searchTerm)
        {
            if (string.IsNullOrWhiteSpace(searchTerm))
            {
                return View("NoResults");
            }

            var newsItem = context.News
                .Where(n => n.Title != null && n.Title.Contains(searchTerm))
                .ToList();

            if (!newsItem.Any())
            {
                return View("NoResults");
            }

            return View(newsItem);
        }
        //..................................................................................................................Edit
        [Authorize(Roles = "Admin , Publisher")]
            public IActionResult Edit(int id)
            {
                ViewBag.cats = context.Categories.ToList();
                var nes = context.News.FirstOrDefault(n => n.Id == id);
                return View(nes);
            }

            [HttpPost]
            public IActionResult SaveEdit(int id, News newNews)
            {
                var nes = context.News.FirstOrDefault(c => c.Id == id);

                if (nes != null)
                {
                    nes.Title = newNews.Title;
                    nes.Content = newNews.Content;
                    nes.CategoryId = newNews.CategoryId;
                    nes.IsFake = newNews.IsFake;
                    nes.Translation = newNews.Translation;
                    nes.image = newNews.image;
                context.SaveChanges();
                }
                return RedirectToAction("NewsAdmin", "News");
            }
            //................................................................................................................Delete
            [Authorize(Roles = "Admin , Publisher")]
            public IActionResult Delete(int id)
            {
                var ne = context.News.FirstOrDefault(s => s.Id == id);
                if (ne != null)
                {
                    context.News.Remove(ne);
                    context.SaveChanges();
                }
                return RedirectToAction("NewsAdmin");
            }
            //................................................................................................................NewsAdmin
            public IActionResult NewsAdmin()
            {
                var cateModel = context.News.ToList();
                return View(cateModel);
            }

        //...........................................................................####.....................................Interact
        [HttpPost]
        public async Task<IActionResult> GetComments(int newsId)
        {
            var comments = await context.Comments
                .Where(c => c.NewsId == newsId)
                .OrderByDescending(c => c.CreatedDate)
                .ToListAsync();

            return PartialView("~/Views/News/GetComments.cshtml", comments);
        }

        // ✅ Post a comment
        [HttpPost]
        public async Task<IActionResult> PostComment(int newsId, string content)
        {
            var user = await userManager.GetUserAsync(User);
            if (user == null || string.IsNullOrWhiteSpace(content))
                return BadRequest();

            var comment = new Comment
            {
                NewsId = newsId,
                Content = content,
                UserId = user.Id,
                CreatedDate = DateTime.Now
            };

            context.Comments.Add(comment);
            await context.SaveChangesAsync();

            return RedirectToAction("Details", new { id = newsId });
        }

        // ✅ Like/Unlike a news item
        [HttpPost]
        public async Task<IActionResult> ToggleLike(int newsId)
        {
            var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (string.IsNullOrEmpty(userId)) return Unauthorized();

            var existingLike = await context.Likes
                .FirstOrDefaultAsync(l => l.NewsId == newsId && l.UserId == userId);

            if (existingLike != null)
            {
                context.Likes.Remove(existingLike);
            }
            else
            {
                context.Likes.Add(new Like
                {
                    NewsId = newsId,
                    UserId = userId,
                    CreatedDate = DateTime.Now
                });
            }

            await context.SaveChangesAsync();
            return Redirect(Request.Headers["Referer"].ToString().ToString());
        }



        // ✅ Get total likes for a news item
        [HttpGet]
        public async Task<IActionResult> GetLikesCount(int newsId)
        {
            int count = await context.Likes.CountAsync(l => l.NewsId == newsId);
            return Json(new { count });
        }


        //---------------------------------------------------------------------------------------------------------


        public IActionResult Details(int id)
        {
            var news = context.News
                .Include(n => n.Comments)
                .ThenInclude(c => c.User) // جلب بيانات المستخدم لكل تعليق
                .FirstOrDefault(n => n.Id == id);

            if (news == null)
                return NotFound();

            // تسجيل قراءة الخبر في سجل المستخدم
            var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (!string.IsNullOrEmpty(userId))
            {
                var history = new UserHistory
                {
                    UserId = userId,
                    NewsId = id.ToString(),
                    ReadAt = DateTime.Now,
                    NewsTitle = news.Title,
                    NewsContent = news.Content
                };

                context.UserHistories.Add(history);
                context.SaveChanges();
            }

            return View(news);
        }

        //==================================================================================

        public async Task<IActionResult> RecommendedNews()
        {
            string userId = User.FindFirstValue(ClaimTypes.NameIdentifier);

            var requestData = new
            {
                user_id = userId,
                top_k = 10
            };

            using var client = new HttpClient();
            var content = new StringContent(JsonSerializer.Serialize(requestData), Encoding.UTF8, "application/json");

            try
            {
                var response = await client.PostAsync("http://localhost:8500/recommend", content);

                if (!response.IsSuccessStatusCode)
                {
                    ViewBag.Error = "لم يتم تحميل التوصيات.";
                    return View(new List<dynamic>());
                }

                var json = await response.Content.ReadAsStringAsync();
                var result = JsonDocument.Parse(json);

                var recommendations = result.RootElement.GetProperty("recommendations")
                    .EnumerateArray()
                    .Select(item => new
                    {
                        NewsId = item.GetProperty("news_id").GetString(),
                        Title = item.GetProperty("title").GetString(),
                        Abstract = item.GetProperty("abstract").GetString(),
                        image = item.TryGetProperty("image", out var img) ? img.GetString() : ""
                    }).ToList<dynamic>();

                return View(recommendations);
            }
            catch (Exception ex)
            {
                ViewBag.Error = "حدث خطأ أثناء جلب التوصيات: " + ex.Message;
                return View(new List<dynamic>());
            }
        }

        [HttpGet]
        public IActionResult FetchNewsFromRss() => View();

        [HttpPost, ActionName("FetchNewsFromRss")]
        public async Task<IActionResult> FetchNewsFromRssPost()
        {
            var rssUrls = new[]
            {
                "https://arabic.cnn.com/rss",
                "https://arabic.cnn.com/rss/sections/business.rss",
                "https://arabic.cnn.com/rss/sections/technology.rss",
                "https://arabic.cnn.com/rss/sections/world.rss",
                "https://arabic.cnn.com/rss/sections/politics.rss",
                "https://arabic.cnn.com/rss/sections/sports.rss",
                "https://feeds.bbci.co.uk/arabic/rss.xml",
                "https://feeds.bbci.co.uk/arabic/rss/business/rss.xml",
                "https://feeds.bbci.co.uk/arabic/rss/world/rss.xml",
                "https://feeds.bbci.co.uk/arabic/rss/science_and_environment/rss.xml",
                "https://feeds.bbci.co.uk/arabic/rss/technology/rss.xml",
                "https://feeds.bbci.co.uk/arabic/rss/sport/rss.xml",
                "https://www.aljazeera.net/aljazeerarss",
                "https://www.aljazeera.net/RssPortal/sections/01e7f06c-7783-41f3-937e-69aed0fadb41/Article",
                "https://www.aljazeera.net/RssPortal/sections/693cc570-5930-4a2b-9fe4-6b06d25a4b29/Article",
                "https://www.aljazeera.net/RssPortal/sections/f3e472ec-6baf-4a7a-b801-af616e1f4a61/Article",
                "https://www.aljazeera.net/RssPortal/sections/092f2a70-bb79-4ec1-9627-032e0eb4a95b/Article",
                "https://www.alarabiya.net/.mrss/ar/rss.xml",
                "https://english.alarabiya.net/rss/ebusiness.xml",
                "https://english.alarabiya.net/rss/esports.xml",
                "https://english.alarabiya.net/rss/technology.xml",
            };

            var defaultCat = context.Categories.FirstOrDefault(c => c.Name == "عام");
            int defaultCatId = defaultCat?.Id ?? context.Categories.Select(c => c.Id).First();

            var lastFetched = context.News
                .Where(n => n.CategoryId == defaultCatId)
                .AsEnumerable() // ⬅️ هذا ينقل التنفيذ من SQL إلى الذاكرة
                .Select(n => n.CreatedAt ?? DateTime.MinValue)
                .DefaultIfEmpty(DateTime.MinValue)
                .Max();

            var allItems = new List<SyndicationItem>();
            foreach (var url in rssUrls)
            {
                try
                {
                    using var reader = XmlReader.Create(url);
                    var feed = SyndicationFeed.Load(reader);
                    if (feed?.Items != null)
                        allItems.AddRange(feed.Items);
                }
                catch { }
            }

            DateTime sevenDaysAgo = DateTime.UtcNow.AddDays(-7);

            var distinct = allItems
                .GroupBy(i => i.Title.Text.Trim())
                .Select(g => g.First())
                .ToList();

            var toAdd = distinct
                .Where(item =>
                {
                    var pub = item.PublishDate.UtcDateTime;
                    if (pub <= lastFetched) return false;
                    var title = item.Title.Text.Trim();
                    return !context.News.Any(n => n.Title == title);
                })
                .ToList();

            var newNewsList = new List<News>();

            foreach (var item in toAdd)
            {
                var title = item.Title.Text.Trim();
                var content = item.Summary?.Text?.Trim() ?? "";
                var image = await ExtractAndDownloadImageAsync(item);

                var news = new News
                {
                    Title = title,
                    Content = content,
                    image = image,
                    CreatedAt = item.PublishDate.UtcDateTime,
                    UpdatedAt = DateTime.UtcNow,
                    IsFake = false,
                    Translation = "------",
                    CategoryId = defaultCatId
                };
                context.News.Add(news);
                newNewsList.Add(news);
            }

            await context.SaveChangesAsync();
            foreach (var news in newNewsList)
            {
                await SendToFastApiAsync(news);
            }
            foreach (var news in newNewsList)
            {
                await SummarizeNewsAsync(news);
            }

            TempData["msg"] = $"✅ تم إضافة {newNewsList.Count} أخبار جديدة من {rssUrls.Length} مصدر RSS";
            return RedirectToAction(nameof(FetchNewsFromRss));
        }

        private async Task<string> DownloadAndSaveImageAsync(string imageUrl)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(imageUrl) || !imageUrl.StartsWith("http"))
                    return null;

                using var httpClient = new HttpClient();
                var bytes = await httpClient.GetByteArrayAsync(imageUrl);
                var extension = Path.GetExtension(imageUrl);
                if (string.IsNullOrEmpty(extension)) extension = ".jpg";
                var fileName = $"{Guid.NewGuid()}{extension}";
                var imagePath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "images", fileName);
                await System.IO.File.WriteAllBytesAsync(imagePath, bytes);

                return fileName;
            }
            catch
            {
                return null;
            }
        }

        private async Task<string> ExtractAndDownloadImageAsync(SyndicationItem item)
        {
            string? imageUrl = null;

            foreach (var ext in item.ElementExtensions)
            {
                using var reader = ext.GetReader();
                if ((reader.LocalName == "thumbnail" || reader.LocalName == "content") && reader.NamespaceURI.Contains("media"))
                {
                    var xe = XElement.Load(reader);
                    var url = xe.Attribute("url")?.Value;
                    if (!string.IsNullOrEmpty(url))
                    {
                        imageUrl = url;
                        break;
                    }
                }
            }

            if (imageUrl == null)
            {
                var enc = item.Links.FirstOrDefault(l =>
                    l.RelationshipType == "enclosure" &&
                    l.MediaType?.StartsWith("image") == true);
                if (enc != null) imageUrl = enc.Uri.ToString();
            }

            if (imageUrl == null)
            {
                var altLink = item.Links.FirstOrDefault(l => l.RelationshipType == "alternate")?.Uri.ToString();
                if (!string.IsNullOrEmpty(altLink))
                {
                    try
                    {
                        var html = await _httpClient.GetStringAsync(altLink);
                        var doc = new HtmlAgilityPack.HtmlDocument();
                        doc.LoadHtml(html);
                        var meta = doc.DocumentNode.SelectSingleNode("//meta[@property='og:image']")
                                ?? doc.DocumentNode.SelectSingleNode("//meta[@name='og:image']");
                        var og = meta?.GetAttributeValue("content", "") ?? "";
                        if (!string.IsNullOrEmpty(og)) imageUrl = og;
                    }
                    catch { }
                }
            }

            if (imageUrl == null)
            {
                var htmlDesc = item.Summary?.Text ?? "";
                var m = Regex.Match(htmlDesc,
                    "<img[^>]+src=[\"']([^\"']+)[\"']",
                    RegexOptions.IgnoreCase);
                if (m.Success) imageUrl = m.Groups[1].Value;
            }
            if (string.IsNullOrWhiteSpace(imageUrl))
            {
                return "default-news.png";  // أو null أو قيمة افتراضية حسب منطقك
            }
            return await DownloadAndSaveImageAsync(imageUrl);
        }

        private async Task SendToFastApiAsync(News news)
        {
            var payload = new
            {
                news_id = news.Id.ToString(),
                title = news.Title,
                content = news.Content,
                image = news.image
            };

            var httpContent = new StringContent(
                JsonSerializer.Serialize(payload),
                System.Text.Encoding.UTF8, "application/json");

            var resp = await _httpClient.PostAsync("http://localhost:8500/api/add_news_full", httpContent);

            Console.WriteLine($"[FAISS] Sent {news.Id}, status {resp.StatusCode}");
        }
        public async Task<IActionResult> ClassifyUncategorizedNews()
        {
            var uncategorizedNews = context.News
                .Where(n => n.CategoryId == null)
                .ToList();

            var httpClient = new HttpClient();

            foreach (var news in uncategorizedNews)
            {
                try
                {
                    string fullText = $"{news.Title} {news.Content}";
                    var response = await httpClient.PostAsJsonAsync("http://localhost:8500/classify", new { text = fullText });

                    if (!response.IsSuccessStatusCode)
                    {
                        _logger.LogWarning($"⚠️ فشل الاتصال بـ API التصنيف للخبر ID={news.Id}");
                        continue;
                    }

                    var result = await response.Content.ReadFromJsonAsync<ClassificationResponse>();
                    if (result == null || string.IsNullOrWhiteSpace(result.Label))
                    {
                        _logger.LogWarning($"⚠️ لم يتمكن من قراءة الاستجابة أو التصنيف فارغ للخبر ID={news.Id}");
                        continue;
                    }

                    string predictedCategory = result.Label.Trim();

                    // التأكد من وجود التصنيف أو إضافته
                    var category = context.Categories.FirstOrDefault(c => c.Name == predictedCategory);
                    if (category == null)
                    {
                        category = new Category { Name = predictedCategory };
                        context.Categories.Add(category);
                        await context.SaveChangesAsync(); // لحفظ التصنيف الجديد قبل استخدامه
                    }

                    // تحديث الخبر بالتصنيف الجديد
                    news.CategoryId = category.Id;
                    context.News.Update(news);
                }
                catch (Exception ex)
                {
                    _logger.LogError($"❌ خطأ أثناء تصنيف الخبر ID={news.Id}: {ex.Message}");
                    continue;
                }
            }

            await context.SaveChangesAsync();

            TempData["msg"] = "✅ تم تصنيف الأخبار بنجاح.";
            return RedirectToAction("Index", "News");
        }


        public class ClassificationResponse
        {
            public string? Label { get; set; }
        }
        public IActionResult ByCategory(int id)
        {
            var category = context.Categories.FirstOrDefault(c => c.Id == id);
            if (category == null) return NotFound();

            var news = context.News
                        .Where(n => n.CategoryId == id)
                        .OrderByDescending(n => n.CreatedAt)
                        .ToList();

            ViewBag.CategoryName = category.Name;
            return View("NewsByCategory", news);
        }
        [HttpPost]
        public async Task<IActionResult> ToggleLikeAjax(int newsId)
        {
            var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (string.IsNullOrEmpty(userId))
                return Unauthorized();

            var existingLike = await context.Likes
                .FirstOrDefaultAsync(l => l.NewsId == newsId && l.UserId == userId);

            if (existingLike != null)
            {
                context.Likes.Remove(existingLike);
            }
            else
            {
                context.Likes.Add(new Like
                {
                    NewsId = newsId,
                    UserId = userId,
                    CreatedDate = DateTime.Now
                });
            }

            await context.SaveChangesAsync();

            var likeCount = await context.Likes.CountAsync(l => l.NewsId == newsId);
            return Json(new { likeCount });
        }


        public class SummarizationResponse
        {
            public int news_id { get; set; }
            public string? Abstract { get; set; }
        }
        private async Task SummarizeNewsAsync(News news)
        {
            var payload = new
            {
                news_id = news.Id,
                content = news.Content,
            };

            var httpContent = new StringContent(
                JsonSerializer.Serialize(payload),
                Encoding.UTF8,
                "application/json"
                );
            try
            {
                var response = await _httpClient.PostAsync("http://localhost:8500/summarize", httpContent);
                if (response.IsSuccessStatusCode)
                {
                    var result =await response.Content.ReadFromJsonAsync<SummarizationResponse>();
                    if(!string.IsNullOrEmpty(result?.Abstract))
                    {
                        news.Abstract = result.Abstract;
                        context.News.Update(news);
                        await context.SaveChangesAsync();
                        Console.WriteLine(" summarized news is done ");
                    }
                }
                else
                {
                    Console.WriteLine("failed to summarize news");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("error in summarize news{ ex.Message}");
            }

        }

        [HttpPost]
        [Authorize(Roles = "Admin")]
        public async Task<IActionResult> SummarizeOldNews()
        {
            var newsWithoutSummary = context.News
                .Where(n => string.IsNullOrEmpty(n.Abstract))
                .ToList();

            int successCount = 0;
            int failCount = 0;

            foreach (var news in newsWithoutSummary)
            {
                try
                {
                    var payload = new
                    {
                        news_id = news.Id,
                        content = news.Content
                    };

                    var httpContent = new StringContent(
                        JsonSerializer.Serialize(payload),
                        Encoding.UTF8,
                        "application/json");

                    var response = await _httpClient.PostAsync("http://localhost:8500/summarize", httpContent);

                    if (response.IsSuccessStatusCode)
                    {
                        var result = await response.Content.ReadFromJsonAsync<SummarizationResponse>();
                        if (!string.IsNullOrEmpty(result?.Abstract))
                        {
                            news.Abstract = result.Abstract;
                            context.News.Update(news);
                            successCount++;
                        }
                        else
                        {
                            failCount++;
                        }
                    }
                    else
                    {
                        failCount++;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ خطأ أثناء تلخيص الخبر ID={news.Id}: {ex.Message}");
                    failCount++;
                }
            }

            await context.SaveChangesAsync();

            TempData["msg"] = $"✅ تم تلخيص {successCount} خبر. ❌ فشل في {failCount} خبر.";
            return RedirectToAction("NewsAdmin");
        }



    }
}

