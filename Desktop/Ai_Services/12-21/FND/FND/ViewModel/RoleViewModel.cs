using System.ComponentModel.DataAnnotations;

namespace FND.ViewModel
{
    public class RoleViewModel
    {
        [Required]
        public string RoleName { get; set; }
    }
}
