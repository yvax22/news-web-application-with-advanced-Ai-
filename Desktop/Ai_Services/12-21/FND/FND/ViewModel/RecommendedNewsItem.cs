using Microsoft.AspNetCore.Mvc.Rendering;
using System.Collections.Generic;

namespace FND.Models.ViewModels
{
    public class RecommendedNewsItem
    {
        public int NewsId { get; set; }
        public string Title { get; set; }
        public string Abstract { get; set; }
        public string Image { get; set; }
    }

}