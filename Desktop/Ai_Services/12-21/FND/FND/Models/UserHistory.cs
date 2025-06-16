using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace FND.Models
{
    public class UserHistory
    {
        [Key]
        public int Id { get; set; }

        [Required]
        public string UserId { get; set; }

        [Required]
        public string NewsId { get; set; }

        [Required]
        public DateTime ReadAt { get; set; }

        public string NewsTitle { get; set; }  // гАлоМо
        public string NewsContent { get; set; } // гАлоМо

        [ForeignKey("UserId")]
        public virtual ApplicationUser User { get; set; }
    }

}
