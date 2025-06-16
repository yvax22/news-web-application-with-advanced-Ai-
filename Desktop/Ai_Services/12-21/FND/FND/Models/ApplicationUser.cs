using FND.Models;
using Microsoft.AspNetCore.Identity;

namespace FND.Models
{
    public class ApplicationUser : IdentityUser
    {

        public string? Address { get; set; }
        public bool IsActive { get; set; } = true;
        public ICollection<News> News { get; set; } = new List<News>();
        public ICollection<Comment> Comments { get; set; } = new List<Comment>();
        public ICollection<Like> Interactions { get; set; } = new List<Like>();

    }
}