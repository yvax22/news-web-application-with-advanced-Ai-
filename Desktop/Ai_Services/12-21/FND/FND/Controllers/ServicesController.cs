using Microsoft.AspNetCore.Mvc;

namespace FND.Controllers
{
    public class ServicesController : Controller
    {
        IWebHostEnvironment webHostEnvironment;

        public ServicesController(IWebHostEnvironment webHostEnvironment)//injection controller
        {
            this.webHostEnvironment = webHostEnvironment;
        }
        public IActionResult uploadImage()
        {
            return View();
        }
        [HttpPost]//Save
        public IActionResult uploadImage(IFormFile Image)
        {
            if (Image == null || Image.Length == 0)
            {

                return RedirectToAction("New", "News");
            }

            string uploadsFolder = Path.Combine(webHostEnvironment.WebRootPath, "Images");
            string uniqueFileName =  Image.FileName;
            string filePath = Path.Combine(uploadsFolder, uniqueFileName);

            using (var fileStream = new FileStream(filePath, FileMode.Create))
            {
                Image.CopyTo(fileStream);
                fileStream.Close();
            }


            return RedirectToAction("New", "News");
        }
        
    }
}
