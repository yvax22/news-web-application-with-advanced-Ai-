using FND.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Security.Claims;
using Microsoft.EntityFrameworkCore;  

namespace FND.Controllers
{
    [Authorize]
    public class HomeController : Controller
    {
        private readonly RecommendationService _recommendationService;
        private readonly Entity _context;

        public HomeController(Entity context, RecommendationService recommendationService)
        {
            _context = context;
            _recommendationService = recommendationService;
        }

        public async Task<IActionResult> Index()
        {
            var latestNews = await _context.News
                .OrderByDescending(n => n.CreatedAt)
                .Take(3)
                .ToListAsync();

            return View(latestNews);
        }


        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }

        
    }
}