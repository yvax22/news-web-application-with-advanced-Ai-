using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore.Metadata.Internal;
using Microsoft.VisualBasic;
using FND.ViewModel;

namespace FND.Controllers
{
    public class RolesController : Controller
    {
        private readonly RoleManager<IdentityRole> roleManager;

        public RolesController(RoleManager<IdentityRole> roleManager)
        {
            this.roleManager = roleManager;
        }
        //Create Role
        //Link
        [HttpGet]
        public IActionResult New()
        {
            return View();
        }
        //Submit
        [HttpPost]
        public async Task<IActionResult> New(RoleViewModel newRoleVM)
        {
            if (ModelState.IsValid)
            {
                IdentityRole role = new IdentityRole();
                role.Name = newRoleVM.RoleName;
                IdentityResult result = await roleManager.CreateAsync(role);
                if (result.Succeeded)
                {
                    return View(new RoleViewModel());
                }
                foreach (var item in result.Errors)
                {
                    //page , save in database
                    ModelState.AddModelError("", item.Description);
                }
            }
            return View(newRoleVM);
        }
    }
}
