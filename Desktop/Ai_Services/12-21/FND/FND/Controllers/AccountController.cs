using FND.Models;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Security.Claims;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Identity;
using FND.ViewModel;
using Microsoft.AspNetCore.Mvc.Rendering;

namespace FND.Controllers
{
    public class AccountController : Controller
    {
        private readonly UserManager<ApplicationUser> userManager;
        private readonly SignInManager<ApplicationUser> signInManager;
        private readonly Entity _context;

        public AccountController(
            UserManager<ApplicationUser> _UserManager,
            SignInManager<ApplicationUser> _SignInManager,
            Entity context)
        {
            userManager = _UserManager;
            signInManager = _SignInManager;
            _context = context; // ✅ DbContext جاهز للاستخدام
        }

        // ============================ Register ============================
        [HttpGet]
        [AllowAnonymous]
        public IActionResult Register()
        {
            return View();
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        [AllowAnonymous]
        public async Task<IActionResult> Register(RegisterUserViewModel newUserVM)
        {
            if (ModelState.IsValid)
            {
                var existingUser = await userManager.FindByNameAsync(newUserVM.Name);
                if (existingUser != null)
                {
                    ModelState.AddModelError("", "اسم المستخدم موجود مسبقًا.");
                    return View(newUserVM);
                }

                var userModel = new ApplicationUser
                {
                    UserName = newUserVM.Name,
                    Address = newUserVM.Address
                };

                var result = await userManager.CreateAsync(userModel, newUserVM.Password);

                if (result.Succeeded)
                {
                    await signInManager.SignInAsync(userModel, false);
                    return RedirectToAction("AllNews", "News");
                }

                foreach (var item in result.Errors)
                {
                    ModelState.AddModelError("", item.Description);
                }
            }

            return View(newUserVM);
        }

        // ============================ Login ============================
        [HttpGet]
        [AllowAnonymous]
        public IActionResult Login()
        {
            return View();
        }

        [HttpPost]
        [AllowAnonymous]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Login(LoginViewModel model, string? returnUrl = null)
        {
            if (ModelState.IsValid)
            {
                var result = await signInManager.PasswordSignInAsync(model.Name, model.Password, model.RememberMe, false);
                if (result.Succeeded)
                {
                    if (!string.IsNullOrEmpty(returnUrl) && Url.IsLocalUrl(returnUrl))
                        return Redirect(returnUrl);

                    return RedirectToAction("Index", "Home");
                }

                ModelState.AddModelError(string.Empty, "اسم المستخدم أو كلمة المرور غير صحيحة");
            }

            return View(model);
        }

        // ============================ Logout ============================
        public async Task<IActionResult> Logout()
        {
            await signInManager.SignOutAsync();
            return RedirectToAction("Login", "Account");
        }

        // ============================ AddPublisher ============================
        [HttpGet]
        public IActionResult AddPublisher()
        {
            var model = new AssignPublisherViewModel
            {
                Users = _context.Users.Select(u => new SelectListItem
                {
                    Value = u.Id,
                    Text = u.UserName
                }).ToList()
            };

            return View(model);
        }


        [HttpPost]
        public async Task<IActionResult> AddPublisher(AssignPublisherViewModel model)
        {
            var user = await userManager.FindByIdAsync(model.SelectedUserId);
            if (user == null)
            {
                ModelState.AddModelError("", "المستخدم غير موجود.");
                return View(model);
            }

            var result = await userManager.AddToRoleAsync(user, "Publisher");

            if (result.Succeeded)
            {
                TempData["msg"] = "✅ تم تعيين المستخدم كناشر.";
                return RedirectToAction("AllNews", "News");
            }

            foreach (var error in result.Errors)
            {
                ModelState.AddModelError("", error.Description);
            }

            return View(model);
        }


        // ============================ AddAdmin ============================
        public IActionResult AddAdmin()
        {
            var model = new AssignAdminViewModel
            {
                Users = _context.Users.Select(u => new SelectListItem
                {
                    Value = u.Id,
                    Text = u.UserName
                }).ToList()
            };

            return View(model);
        }


        [HttpPost]
        public async Task<IActionResult> AddAdmin(AssignAdminViewModel model)
        {
            var user = await userManager.FindByIdAsync(model.SelectedUserId);
            if (user == null)
            {
                ModelState.AddModelError("", "المستخدم غير موجود.");
                return View(model);
            }

            var result = await userManager.AddToRoleAsync(user, "Admin");

            if (result.Succeeded)
            {
                TempData["msg"] = "✅ تم تعيين المستخدم كمسؤول.";
                return RedirectToAction("AllNews", "News");
            }

            foreach (var error in result.Errors)
            {
                ModelState.AddModelError("", error.Description);
            }

            return View(model);
        }

    }
}
