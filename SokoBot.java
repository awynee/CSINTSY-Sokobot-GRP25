package solver;

import java.awt.Point;
import java.util.HashSet;
import java.util.Set;
import java.util.PriorityQueue;
import java.util.Comparator;

public class SokoBot {

    // Represent a game state
    class State {
        Point player;
        Set<Point> boxes;
        String path;
        int cost;
        int heuristic;
        int total;

        State(Point player, Set<Point> boxes, String path, int cost, int heuristic, boolean greedy) {
            this.player = player;
            this.boxes = boxes;
            this.path = path;
            this.cost = cost;
            this.heuristic = heuristic;
            // A* uses cost + heuristic; Greedy BFS uses only heuristic
            this.total = greedy ? heuristic : cost + heuristic;
        }
    }

    public String solveSokobanPuzzle(int width, int height, char[][] mapData, char[][] itemsData) {
        
    Point player = null;
    Set<Point> boxes = new HashSet<>();
    Set<Point> targets = new HashSet<>();

    // Scan the map and record player, boxes, and targets
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

    // Decide which algorithm to use based on number of boxes
    boolean greedyMode = boxes.size() > 5;

    // comparator for lowest total cost (A* priority)
    // Priority queue for lowest total cost (A* or GBFS priority)
    PriorityQueue<State> open = new PriorityQueue<>(Comparator.comparingInt(s -> s.total));
    open.add(new State(player, boxes, "", 0, heuristic(boxes, targets), greedyMode));

    Set<String> visited = new HashSet<>();

    long startTime = System.currentTimeMillis();
    int explored = 0;

    // directions
    char[] moves = {'u', 'd', 'l', 'r'};
    int[] dx = {0, 0, -1, 1};
    int[] dy = {-1, 1, 0, 0};

    // main loop
    while (!open.isEmpty()) {
        if (System.currentTimeMillis() - startTime > 15000) {
            System.out.println("Timeout (15 seconds). Explored states: " + explored); // FOR DEBUGGING - CAN REMOVE
            return "";
        }

        State current = open.poll();
        explored++;

        // unique key to detect repeated states
        String key = makeKey(current.player, current.boxes);
        if (visited.contains(key))
            continue;
        visited.add(key);

        // goal condition: all boxes are on target
        if (targets.containsAll(current.boxes)) {
            System.out.println("Moves: " + current.path);
            return current.path;
        }

        // generate new states (try moves)
        for (int i = 0; i < 4; i++) {
            State next = tryMove(current, moves[i], dx[i], dy[i], mapData, targets, greedyMode);
            if (next != null)
                open.add(next);
            }
        }
        
        return "";
    }

    // try moving player (and possibly pushing a box)
    private State tryMove(State current, char move, int dx, int dy, char[][] mapData, Set<Point> targets, boolean greedy) {
        Point newPlayer = new Point(current.player.x + dx, current.player.y + dy);

        // wall check
        if (mapData[newPlayer.y][newPlayer.x] == '#')
            return null;

        // deep copy boxes
        Set<Point> newBoxes = new HashSet<>();
        for (Point b : current.boxes)
            newBoxes.add(new Point(b));

        // check if player is moving into a box
        if (newBoxes.contains(newPlayer)) {
            Point boxNew = new Point(newPlayer.x + dx, newPlayer.y + dy);

            // Blocked by wall or another box
            if (mapData[boxNew.y][boxNew.x] == '#' || newBoxes.contains(boxNew))
                return null;

            // move the box
            newBoxes.remove(newPlayer);
            newBoxes.add(boxNew);
        }

        // Deadlock check
        if (isDeadlocked(newBoxes, mapData))
            return null;

        // return new state
        return new State(newPlayer, newBoxes, current.path + move,
                current.cost + 1, heuristic(newBoxes, targets), greedy);
    }

    // Manhattan distance heuristic
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

    // checking if a corner is a dead position
    private boolean isDeadlocked(Set<Point> boxes, char[][] map) {
        int height = map.length;
        int width = map[0].length;

        for (Point b : boxes) {
            if (map[b.y][b.x] == '.') continue; // target = okay

            boolean wallLeft = (b.x - 1 < 0) || (map[b.y][b.x - 1] == '#');
            boolean wallRight = (b.x + 1 >= width) || (map[b.y][b.x + 1] == '#');
            boolean wallUp = (b.y - 1 < 0) || (map[b.y - 1][b.x] == '#');
            boolean wallDown = (b.y + 1 >= height) || (map[b.y + 1][b.x] == '#');

            if ((wallLeft && wallUp) || (wallLeft && wallDown)
                    || (wallRight && wallUp) || (wallRight && wallDown))
                return true;
        }
        return false;
    }

    // makes a key for every state and is stored in a string to avoid duplicates
    private String makeKey(Point player, Set<Point> boxes) {
        StringBuilder sb = new StringBuilder();
        sb.append(player.x).append(',').append(player.y).append('|');
        boxes.stream()
                .sorted((a, b) -> a.x == b.x ? a.y - b.y : a.x - b.x)
                .forEach(p -> sb.append(p.x).append(',').append(p.y).append(';'));
        return sb.toString();
    }
}
