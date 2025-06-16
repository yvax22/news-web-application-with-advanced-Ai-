using FND.Models;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;

namespace FND.Controllers
{
    public class UsersController : Controller
    {
        private readonly Entity _db;

        public UsersController(Entity db)
        {
            _db = db;
        }
        public IActionResult Users()
        {
            
                List<ApplicationUser> AppModel = _db.Users.ToList();

                return View(AppModel);
            
        }
        
    }
}
