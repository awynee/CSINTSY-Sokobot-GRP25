package solver;

import java.awt.Point;
import java.util.HashSet;
import java.util.Set;
import java.util.PriorityQueue;
import java.util.Comparator;

public class SokoBot {

	// represent a game state
    class State {
      Point player;
      Set<Point> boxes;
      String path;
      int cost;
      int heuristic;
      int total;

      State(Point player, Set<Point> boxes, String path, int cost, int heuristic) {
        this.player = player;
        this.boxes = boxes;
        this.path = path;
        this.cost = cost;
        this.heuristic = heuristic;
        this.total = cost + heuristic;
      }
    }

    public String solveSokobanPuzzle(int width, int height, char[][] mapData, char[][] itemsData) {

    Point player = null;
    Set<Point> boxes = new HashSet<>();
    Set<Point> targets = new HashSet<>();

    // scan the map and record player, boxes, and targets
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        if (mapData[i][j] == '.') // target
          targets.add(new Point(j, i));
        if (itemsData[i][j] == '$') // box
          boxes.add(new Point(j, i));
        if (itemsData[i][j] == '@') // player
          player = new Point(j, i);
      }
    }

    // comparator for lowest total cost (A* priority)
    PriorityQueue<State> open = new PriorityQueue<>(Comparator.comparingInt(s -> s.total));
    open.add(new State(player, boxes, "", 0, heuristic(boxes, targets)));

    Set<String> visited = new HashSet<>();

    long startTime = System.currentTimeMillis();

    // directions
    char[] moves = { 'u', 'd', 'l', 'r' };
    int[] dx = { 0, 0, -1, 1 };
    int[] dy = { -1, 1, 0, 0 };

    // a* loop
    while (!open.isEmpty()) {
      State current = open.poll();

      // unique key to detect repeated states
      String key = current.player.x + "," + current.player.y + current.boxes.toString();
      if (visited.contains(key))
        continue;
      visited.add(key);

      // goal condition: all boxes are on targets
      if (targets.containsAll(current.boxes))
        return current.path;

      // generate new states (try moves)
      for (int i = 0; i < 4; i++) {
        State next = tryMove(current, moves[i], dx[i], dy[i], mapData, targets);
        if (next != null) {
          open.add(next);
        }
      }
    }

    // No solution found within time
    return "";
  }

  //try moving player (and possibly pushing box)
  private State tryMove(State current, char move, int dx, int dy, char[][] mapData, Set<Point> targets) {
    Point newPlayer = new Point(current.player.x + dx, current.player.y + dy);

    //wall check
    if (mapData[newPlayer.y][newPlayer.x] == '#')
      return null;

    //deep copy of boxes
    Set<Point> newBoxes = new HashSet<>();
    for (Point b : current.boxes)
      newBoxes.add(new Point(b));

    //check if player is moving into a box
    if (newBoxes.contains(newPlayer)) {
      Point boxNew = new Point(newPlayer.x + dx, newPlayer.y + dy);

      // Blocked by wall or another box
      if (mapData[boxNew.y][boxNew.x] == '#' || newBoxes.contains(boxNew))
        return null;

      //move the box
      newBoxes.remove(newPlayer);
      newBoxes.add(boxNew);
    }

    //return new state
    return new State(newPlayer, newBoxes, current.path + move, current.cost + 1, heuristic(newBoxes, targets));
  }

  //manhattan distance heuristic
  private int heuristic(Set<Point> boxes, Set<Point> targets) {
    int total = 0;
    for (Point box : boxes) {
      int best = Integer.MAX_VALUE;
      for (Point goal : targets) {
        int d = Math.abs(box.x - goal.x) + Math.abs(box.y - goal.y);
        if (d < best)
          best = d;
      }
      total += best;
    }
    return total;
  }
}
