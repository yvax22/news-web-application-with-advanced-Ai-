using System.ComponentModel.DataAnnotations;
using Microsoft.AspNetCore.Identity;
using FND.Models;



namespace FND.ViewModel
{
    public class RegisterUserViewModel
    {
        [Required]
        public string Name { get; set; }

        [DataType(DataType.Password)]
        [Required]
        public string Password { get; set; }

        [DataType(DataType.Password)]
        [Required]
        [Compare("Password")] //تقارن
        public string ConfirmPassword { get; set; }

        [Display(Name = "رقم الهاتف")]
        [Phone(ErrorMessage = "الرجاء إدخال رقم هاتف صالح.")]
        public string PhoneNumber { get; set; }
        public string? Address { get; set; } = string.Empty;
    }
}
