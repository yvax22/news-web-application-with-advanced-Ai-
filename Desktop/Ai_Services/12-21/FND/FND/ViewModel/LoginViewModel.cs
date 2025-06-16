using System.ComponentModel.DataAnnotations;

namespace FND.ViewModel
{
    public class LoginViewModel
    {
        [Required]
        public string Name { get; set; }
        [Required]
        [DataType(DataType.Password)]
        public string Password { get; set; }

        public bool RememberMe { get; set; }
    }
}
