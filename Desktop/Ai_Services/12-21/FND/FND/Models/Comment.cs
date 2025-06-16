using FND.Models;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
namespace FND.Models
{
    public class Comment
    {
        [Key]
        public int Id { get; set; } 
        [Required] 
        public string? Content { get; set; }
        public string? UserId { get; set; }
        public int NewsId { get; set; }
        public DateTime CreatedDate { get; set; }

        public virtual ApplicationUser User { get; set; } = new ApplicationUser();
        public virtual News? News { get; set; }
    }
}
