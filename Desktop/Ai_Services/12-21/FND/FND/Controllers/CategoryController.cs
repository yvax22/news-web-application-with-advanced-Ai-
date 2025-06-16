using FND.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace FND.Controllers
{
    //[Authorize]
    public class CategoryController : Controller
    {
        private readonly Entity _db;

        public CategoryController(Entity db)
        {
            _db = db;
        }
        //................................................................................................Index
        public IActionResult Index()
        {
            List<Category> deptList = _db.Categories.ToList();
            return View(deptList);
        }
        //................................................................................................New
        public IActionResult New()
        {
            //view="New",Model=null
            return View(new Category());
        }
        //<form method="post">
        [HttpPost]
        public IActionResult SaveNew(Category cat)//Brimitev Type -> if exist -> Storage -> else = Null .
        {
            if (cat.Name != null)
            {
                _db.Categories.Add(cat);
                _db.SaveChanges();
                return RedirectToAction("Index");
            }
            return View();
        }
        //................................................................................................Details
        public IActionResult Details(int id)
        {
            var cat = _db.Categories
                .Include(c => c.News) // ⬅️ جلب الأخبار المرتبطة
                .FirstOrDefault(c => c.Id == id);

            if (cat == null)
                return NotFound();

            return View("Details", cat);
        }

        //................................................................................................Delete
        public IActionResult Delete(int id)
        {
            Category? cat = _db.Categories.FirstOrDefault(s => s.Id == id);
            if (cat == null)
            {
                return Content("القسم غير موجود");

            }

            _db.Categories.Remove(cat);
            _db.SaveChanges();
            return RedirectToAction("Index");
        }

        //................................................................................................Categories
        public IActionResult Category()
        {
            List<Category> cateModel = _db.Categories.ToList();
            return View(cateModel);
        }
        //................................................................................................Sport
        public IActionResult Sport()
        {
            var category = _db.Categories.FirstOrDefault(c => c.Name == "رياضة");
            if (category == null) return NotFound();
            
            var news = _db.News.Where(n => n.CategoryId == category.Id).ToList();
            ViewBag.CategoryName = "أخبار الرياضة";
            return View("Category", news);
        }
        //................................................................................................Political
        public IActionResult Political()
        {
            var category = _db.Categories.FirstOrDefault(c => c.Name == "سياسة");
            if (category == null) return NotFound();
            
            var news = _db.News.Where(n => n.CategoryId == category.Id).ToList();
            ViewBag.CategoryName = "أخبار السياسة";
            return View("Category", news);
        }

        //................................................................................................Economic
        public IActionResult Economic()
        {
            
            List<News> cateModel = _db.News.ToList();
            

            return View(cateModel);
        }
        //................................................................................................Technology
        public IActionResult Technology()
        {
            var category = _db.Categories.FirstOrDefault(c => c.Name == "تكنولوجيا");
            if (category == null) return NotFound();
            
            var news = _db.News.Where(n => n.CategoryId == category.Id).ToList();
            ViewBag.CategoryName = "أخبار التكنولوجيا";
            return View("Category", news);
        }
        //................................................................................................Food
        public IActionResult Food()
        {
            var category = _db.Categories.FirstOrDefault(c => c.Name == "الطعام والشراب");
            if (category == null) return NotFound();
            
            var news = _db.News.Where(n => n.CategoryId == category.Id).ToList();
            ViewBag.CategoryName = "أخبار الطعام والشراب";
            return View("Category", news);
        }
        //................................................................................................StyleAndBeauty
        public IActionResult StyleAndBeauty()
        {
            var category = _db.Categories.FirstOrDefault(c => c.Name == "الجمال والموضة");
            if (category == null) return NotFound();
            
            var news = _db.News.Where(n => n.CategoryId == category.Id).ToList();
            ViewBag.CategoryName = "أخبار الجمال والموضة";
            return View("Category", news);
        }
        //................................................................................................Education
        public IActionResult Education()
        {
            var category = _db.Categories.FirstOrDefault(c => c.Name == "تعليم");
            if (category == null) return NotFound();
            
            var news = _db.News.Where(n => n.CategoryId == category.Id).ToList();
            ViewBag.CategoryName = "أخبار التعليم";
            return View("Category", news);
        }
        //................................................................................................Crime
        public IActionResult Crime()
        {
            var category = _db.Categories.FirstOrDefault(c => c.Name == "جرائم");
            if (category == null) return NotFound();
            
            var news = _db.News.Where(n => n.CategoryId == category.Id).ToList();
            ViewBag.CategoryName = "أخبار الجرائم";
            return View("Category", news);
        }
    }
}
